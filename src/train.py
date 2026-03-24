#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

from video_dataset import VideoDataset
import asot
from utils import *
from metrics import ClusteringMetrics, indep_eval_metrics

num_eps = 1e-11


def build_mlp(layer_sizes):
    """MLP: [in, h1, h2, ..., out] with ReLU between hidden layers."""
    layers = [
        nn.Sequential(nn.Linear(sz, sz1), nn.ReLU())
        for sz, sz1 in zip(layer_sizes[:-2], layer_sizes[1:-1])
    ]
    layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1])]
    return nn.Sequential(*layers)


class VideoSSL(pl.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        weight_decay=1e-4,
        layer_sizes=[64, 128, 40],
        layer_sizes_txt=None,        # if None, reuse layer_sizes
        n_clusters=20,
        beta_mm=0.8,                 # beta for visual cost
        use_mm_cost=False,           # enable multimodal cost (separate costs)
        alpha_train=0.3,
        alpha_eval=0.3,
        n_ot_train=[50, 1],
        n_ot_eval=[50, 1],
        step_size=None,
        train_eps=0.06,
        eval_eps=0.01,
        ub_frames=False,
        ub_actions=True,
        lambda_frames_train=0.05,
        lambda_actions_train=0.05,
        lambda_frames_eval=0.05,
        lambda_actions_eval=0.01,
        temp=0.1,
        radius_gw=0.04,
        learn_clusters=True,
        n_frames=256,
        rho=0.1,
        exclude_cls=None,
        visualize=False,
    ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay

        self.n_clusters = n_clusters
        self.learn_clusters = learn_clusters
        self.exclude_cls = exclude_cls
        self.visualize = visualize

        self.layer_sizes = layer_sizes
        self.layer_sizes_txt = layer_sizes_txt if layer_sizes_txt is not None else layer_sizes

        # multimodal knobs
        self.beta_mm = beta_mm
        self.use_mm_cost = use_mm_cost

        self.alpha_train = alpha_train
        self.alpha_eval = alpha_eval
        self.n_ot_train = n_ot_train
        self.n_ot_eval = n_ot_eval
        self.step_size = step_size
        self.train_eps = train_eps
        self.eval_eps = eval_eps
        self.radius_gw = radius_gw
        self.ub_frames = ub_frames
        self.ub_actions = ub_actions
        self.lambda_frames_train = lambda_frames_train
        self.lambda_actions_train = lambda_actions_train
        self.lambda_frames_eval = lambda_frames_eval
        self.lambda_actions_eval = lambda_actions_eval

        self.temp = temp
        self.n_frames = n_frames
        self.rho = rho

        # two heads (visual + text)
        self.mlp_vis = build_mlp(self.layer_sizes)
        self.mlp_txt = build_mlp(self.layer_sizes_txt)

        # shared prototypes in projected space
        d_vis = self.layer_sizes[-1]
        d_txt = self.layer_sizes_txt[-1]
        assert d_vis == d_txt, "visual/text heads must output same dim!"
        self.proj_dim = d_vis

        self.clusters = nn.parameter.Parameter(
            data=F.normalize(torch.randn(self.n_clusters, self.proj_dim), dim=-1),
            requires_grad=learn_clusters,
        )

        # metrics
        self.mof = ClusteringMetrics(metric="mof")
        self.f1 = ClusteringMetrics(metric="f1")
        self.miou = ClusteringMetrics(metric="miou")
        self.save_hyperparameters()
        self.test_cache = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    @torch.no_grad()
    def _normalize_clusters(self):
        self.clusters.data = F.normalize(self.clusters.data, dim=-1)

    def _project(self, vis_raw, txt_raw=None):
        """
        Projects raw visual and text features into shared latent space.
        Returns (z_vis, z_txt, z_mix)
        z_mix is always defined (visual-only if txt missing).
        """
        B, T, _ = vis_raw.shape
        D = self.proj_dim

        z_vis = F.normalize(
            self.mlp_vis(vis_raw.reshape(-1, vis_raw.shape[-1])).reshape(B, T, D),
            dim=-1,
        )

        z_txt = None
        if txt_raw is not None:
            assert txt_raw.shape[:2] == (B, T), "txt_raw must match (B,T) of vis_raw"
            z_txt = F.normalize(
                self.mlp_txt(txt_raw.reshape(-1, txt_raw.shape[-1])).reshape(B, T, D),
                dim=-1,
            )

        if z_txt is None:
            z_mix = z_vis
        else:
            z_mix = F.normalize(self.beta_mm * z_vis + (1.0 - self.beta_mm) * z_txt, dim=-1)

        return z_vis, z_txt, z_mix

    def _multimodal_cost(self, z_vis, z_txt, T, device):
        """
          C = beta*C_vis + (1-beta)*C_txt
          C_vis = 1 - cos(z_vis, a)
          C_txt = 1 - cos(z_txt, a)
        """
        self._normalize_clusters()

        C_vis = 1.0 - (z_vis @ self.clusters.T.unsqueeze(0))  # (B,T,K)

        if z_txt is None:
            C = C_vis
        else:
            C_txt = 1.0 - (z_txt @ self.clusters.T.unsqueeze(0))
            C = self.beta_mm * C_vis + (1.0 - self.beta_mm) * C_txt

        temp_prior = asot.temporal_prior(T, self.n_clusters, self.rho, device)
        return C + temp_prior

    def _single_cost(self, z_mix, T, device):
        """Baseline cost using only mixed embedding (or visual-only)."""
        self._normalize_clusters()
        C = 1.0 - (z_mix @ self.clusters.T.unsqueeze(0))
        temp_prior = asot.temporal_prior(T, self.n_clusters, self.rho, device)
        return C + temp_prior

    def _unpack_batch(self, batch):
        """
        Supports:
          (features, mask, gt, fname, nsub)
        where features can be:
          - Tensor (B,T,D)
          - (vis, txt)
        """
        features_raw, mask, gt, fname, n_subactions = batch
        if isinstance(features_raw, (tuple, list)) and len(features_raw) == 2:
            vis_raw, txt_raw = features_raw
        else:
            vis_raw, txt_raw = features_raw, None
        return vis_raw, txt_raw, mask, gt, fname, n_subactions

    def training_step(self, batch, batch_idx):
        vis_raw, txt_raw, mask, gt, fname, _ = self._unpack_batch(batch)
        B, T, _ = vis_raw.shape
        device = vis_raw.device

        z_vis, z_txt, z_mix = self._project(vis_raw, txt_raw)

        # codes for CE: use mixed embedding
        self._normalize_clusters()
        logits = (z_mix @ self.clusters.T[None, ...]) / self.temp
        codes = F.softmax(logits, dim=-1)

        # pseudo-labels from OT
        with torch.no_grad():
            if self.use_mm_cost:
                cost_matrix = self._multimodal_cost(z_vis, z_txt, T, device)
            else:
                cost_matrix = self._single_cost(z_mix, T, device)

            opt_codes, _ = asot.segment_asot(
                cost_matrix,
                mask,
                eps=self.train_eps,
                alpha=self.alpha_train,
                radius=self.radius_gw,
                ub_frames=self.ub_frames,
                ub_actions=self.ub_actions,
                lambda_frames=self.lambda_frames_train,
                lambda_actions=self.lambda_actions_train,
                n_iters=self.n_ot_train,
                step_size=self.step_size,
            )

        loss_ce = -((opt_codes * torch.log(codes + num_eps)) * mask[..., None]).sum(dim=2).mean()
        self.log("train_loss", loss_ce)
        return loss_ce

    def validation_step(self, batch, batch_idx):
        vis_raw, txt_raw, mask, gt, fname, _ = self._unpack_batch(batch)
        B, T, _ = vis_raw.shape
        device = vis_raw.device

        z_vis, z_txt, z_mix = self._project(vis_raw, txt_raw)

        if self.use_mm_cost:
            cost_matrix = self._multimodal_cost(z_vis, z_txt, T, device)
        else:
            cost_matrix = self._single_cost(z_mix, T, device)

        segmentation, _ = asot.segment_asot(
            cost_matrix,
            mask,
            eps=self.eval_eps,
            alpha=self.alpha_eval,
            radius=self.radius_gw,
            ub_frames=self.ub_frames,
            ub_actions=self.ub_actions,
            lambda_frames=self.lambda_frames_eval,
            lambda_actions=self.lambda_actions_eval,
            n_iters=self.n_ot_eval,
            step_size=self.step_size,
        )

        segments = segmentation.argmax(dim=2)
        self.mof.update(segments, gt, mask)
        self.f1.update(segments, gt, mask)
        self.miou.update(segments, gt, mask)

        metrics = indep_eval_metrics(segments, gt, mask, ["mof", "f1", "miou"], exclude_cls=self.exclude_cls)
        self.log("val_mof_per", metrics["mof"])
        self.log("val_f1_per", metrics["f1"])
        self.log("val_miou_per", metrics["miou"])

        # validation loss: CE on pseudo labels (same cost as above)
        self._normalize_clusters()
        logits = (z_mix @ self.clusters.T[None, ...]) / self.temp
        codes = F.softmax(logits, dim=-1)

        pseudo_labels, _ = asot.segment_asot(
            cost_matrix,
            mask,
            eps=self.train_eps,
            alpha=self.alpha_train,
            radius=self.radius_gw,
            ub_frames=self.ub_frames,
            ub_actions=self.ub_actions,
            lambda_frames=self.lambda_frames_train,
            lambda_actions=self.lambda_actions_train,
            n_iters=self.n_ot_train,
            step_size=self.step_size,
        )
        loss_ce = -((pseudo_labels * torch.log(codes + num_eps)) * mask[..., None]).sum(dim=[1, 2]).mean()
        self.log("val_loss", loss_ce)

        spacing = max(1, int(self.trainer.num_val_batches[0] / 5))
        if batch_idx % spacing == 0 and wandb.run is not None and self.visualize:
            plot_idx = int(batch_idx / spacing)
            gt_cpu = gt[0].cpu().numpy()

            fdists = squareform(pdist(z_mix[0].detach().cpu().numpy(), "cosine"))
            fig = plot_matrix(
                fdists,
                gt=gt_cpu,
                colorbar=False,
                title=fname[0],
                figsize=(5, 5),
                xlabel="Frame index",
                ylabel="Frame index",
            )
            wandb.log({f"val_pairwise_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            plt.close()

            fig = plot_matrix(
                codes[0].detach().cpu().numpy().T,
                gt=gt_cpu,
                colorbar=False,
                title=fname[0],
                figsize=(10, 5),
                xlabel="Frame index",
                ylabel="Action index",
            )
            wandb.log({f"val_P_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            plt.close()

            fig = plot_matrix(
                pseudo_labels[0].detach().cpu().numpy().T,
                gt=gt_cpu,
                colorbar=False,
                title=fname[0],
                figsize=(10, 5),
                xlabel="Frame index",
                ylabel="Action index",
            )
            wandb.log({f"val_OT_PL_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            plt.close()

            fig = plot_matrix(
                segmentation[0].detach().cpu().numpy().T,
                gt=gt_cpu,
                colorbar=False,
                title=fname[0],
                figsize=(10, 5),
                xlabel="Frame index",
                ylabel="Action index",
            )
            wandb.log({f"val_OT_pred_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            plt.close()

        return None

    def test_step(self, batch, batch_idx):
        vis_raw, txt_raw, mask, gt, fname, _ = self._unpack_batch(batch)
        B, T, _ = vis_raw.shape
        device = vis_raw.device

        z_vis, z_txt, z_mix = self._project(vis_raw, txt_raw)

        if self.use_mm_cost:
            cost_matrix = self._multimodal_cost(z_vis, z_txt, T, device)
        else:
            cost_matrix = self._single_cost(z_mix, T, device)

        segmentation, _ = asot.segment_asot(
            cost_matrix,
            mask,
            eps=self.eval_eps,
            alpha=self.alpha_eval,
            radius=self.radius_gw,
            ub_frames=self.ub_frames,
            ub_actions=self.ub_actions,
            lambda_frames=self.lambda_frames_eval,
            lambda_actions=self.lambda_actions_eval,
            n_iters=self.n_ot_eval,
            step_size=self.step_size,
        )

        segments = segmentation.argmax(dim=2)
        self.mof.update(segments, gt, mask)
        self.f1.update(segments, gt, mask)
        self.miou.update(segments, gt, mask)

        metrics = indep_eval_metrics(segments, gt, mask, ["mof", "f1", "miou"], exclude_cls=self.exclude_cls)
        self.log("test_mof_per", metrics["mof"])
        self.log("test_f1_per", metrics["f1"])
        self.log("test_miou_per", metrics["miou"])

        self.test_cache.append([metrics["mof"], segments, gt, mask, fname])
        return None

    def on_validation_epoch_end(self):
        mof, pred_to_gt = self.mof.compute(exclude_cls=self.exclude_cls)
        f1, _ = self.f1.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        miou, _ = self.miou.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        self.log("val_mof_full", mof)
        self.log("val_f1_full", f1)
        self.log("val_miou_full", miou)
        self.mof.reset()
        self.f1.reset()
        self.miou.reset()

    def on_test_epoch_end(self):
        mof, pred_to_gt = self.mof.compute(exclude_cls=self.exclude_cls)
        f1, _ = self.f1.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        miou, _ = self.miou.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        self.log("test_mof_full", mof)
        self.log("test_f1_full", f1)
        self.log("test_miou_full", miou)

        if wandb.run is not None and self.visualize:
            for i, (mof_i, pred, gt, mask, fname) in enumerate(self.test_cache):
                self.test_cache[i][0] = indep_eval_metrics(
                    pred, gt, mask, ["mof"], exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt
                )["mof"]
            self.test_cache = sorted(self.test_cache, key=lambda x: x[0], reverse=True)

            for i, (mof_i, pred, gt, mask, fname) in enumerate(self.test_cache[:10]):
                fig = plot_segmentation_gt(
                    gt, pred, mask,
                    exclude_cls=self.exclude_cls,
                    pred_to_gt=pred_to_gt,
                    gt_uniq=np.unique(self.mof.gt_labels),
                    name=f"{fname[0]}",
                )
                wandb.log({f"test_segment_{i}": wandb.Image(fig), "trainer/global_step": self.trainer.global_step})
                plt.close()

        out_dir = Path(self.trainer.log_dir) / "preds_raw"
        out_dir.mkdir(parents=True, exist_ok=True)

        for (mof_i, pred, gt, mask, fname) in self.test_cache:
            vid = Path(fname[0]).stem
            np.savez_compressed(
                out_dir / f"{vid}.npz",
                pred=pred[0].detach().cpu().numpy(),
                gt=gt[0].detach().cpu().numpy(),
                mask=mask[0].detach().cpu().numpy(),
                fname=fname[0],
            )
            
        self.test_cache = []
        self.mof.reset()
        self.f1.reset()
        self.miou.reset()

    def fit_clusters(self, dataloader, K):
        """
        KMeans init on z_mix (stable and consistent with CE logits).
        """
        with torch.no_grad():
            feats_all = []
            self.mlp_vis.eval()
            self.mlp_txt.eval()

            for batch in dataloader:
                vis_raw, txt_raw, mask, gt, fname, _ = self._unpack_batch(batch)
                z_vis, z_txt, z_mix = self._project(vis_raw, txt_raw)
                feats_all.append(z_mix)

            feats_all = torch.cat(feats_all, dim=0).reshape(-1, feats_all[0].shape[-1]).cpu().numpy()
            kmeans = KMeans(n_clusters=K).fit(feats_all)

            self.mlp_vis.train()
            self.mlp_txt.train()

        self.clusters.data = torch.from_numpy(kmeans.cluster_centers_).to(self.clusters.device)
        self._normalize_clusters()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train representation learning pipeline")

    # FUGW OT segmentation parameters
    parser.add_argument("--alpha-train", "-at", type=float, default=0.3)
    parser.add_argument("--alpha-eval", "-ae", type=float, default=0.6)
    parser.add_argument("--ub-frames", "-uf", action="store_true")
    parser.add_argument("--ub-actions", "-ua", action="store_true")
    parser.add_argument("--lambda-frames-train", "-lft", type=float, default=0.05)
    parser.add_argument("--lambda-actions-train", "-lat", type=float, default=0.05)
    parser.add_argument("--lambda-frames-eval", "-lfe", type=float, default=0.05)
    parser.add_argument("--lambda-actions-eval", "-lae", type=float, default=0.01)
    parser.add_argument("--eps-train", "-et", type=float, default=0.07)
    parser.add_argument("--eps-eval", "-ee", type=float, default=0.04)
    parser.add_argument("--radius-gw", "-r", type=float, default=0.04)
    parser.add_argument("--n-ot-train", "-nt", type=int, nargs="+", default=[25, 1])
    parser.add_argument("--n-ot-eval", "-no", type=int, nargs="+", default=[25, 1])
    parser.add_argument("--step-size", "-ss", type=float, default=None)

    # dataset params
    parser.add_argument("--base-path", "-p", type=str, default="/home/users/u6567085/data")
    parser.add_argument("--dataset", "-d", type=str, required=True)
    parser.add_argument("--activity", "-ac", type=str, nargs="+", required=True)
    parser.add_argument("--exclude", "-x", type=int, default=None)
    parser.add_argument("--n-frames", "-f", type=int, default=256)
    parser.add_argument("--std-feats", "-s", action="store_true")

    # NEW: multimodal dirs (THIS fixes your error)
    parser.add_argument(
        "--visual-dir",
        type=str,
        default=None,
        help="Path to visual per-frame .npy features (one file per video). If set with --text-dir, dataset returns (vis, txt).",
    )
    parser.add_argument(
        "--text-dir",
        type=str,
        default=None,
        help="Path to text .npz features (embeddings,start_sec,end_sec). If set with --visual-dir, dataset returns (vis, txt).",
    )

    parser.add_argument(
        "--caption-name",
        type=str,
        default="none",
        help="Short identifier for caption variant (e.g. gemma1, gemma2, SAC, TAA)",
    )


    # fold + split-root
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split-root", type=str, default="splits")

    # feature name for logging
    parser.add_argument("--feature-name", type=str, default="unknown")

    # representation learning params
    parser.add_argument("--n-epochs", "-ne", type=int, default=15)
    parser.add_argument("--batch-size", "-bs", type=int, default=2)
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--k-means", "-km", action="store_false")
    parser.add_argument("--layers", "-ls", default=[64, 128, 40], nargs="+", type=int)

    # optional different text head
    parser.add_argument("--layers-txt", default=None, nargs="+", type=int)

    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--n-clusters", "-c", type=int, default=8)

    # multimodal knobs
    parser.add_argument("--beta-mm", type=float, default=0.5, help="beta for visual cost; (1-beta) for text")
    parser.add_argument("--use-mm-cost", action="store_true", help="enable multimodal cost (beta*Cvis+(1-beta)*Ctxt)")

    # system/logging params
    parser.add_argument("--val-freq", "-vf", type=int, default=5)
    parser.add_argument("--gpu", "-g", type=int, default=1)
    parser.add_argument("--wandb", "-w", action="store_true")
    parser.add_argument("--visualize", "-v", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--group", type=str, default="base")
    args = parser.parse_args()

    # early safety
    if args.use_mm_cost and args.text_dir is None:
        raise ValueError("You passed --use-mm-cost but did not pass --text-dir. Provide text features or remove --use-mm-cost.")

    pl.seed_everything(args.seed)

    # -------------------------
    # Run name (saved inside version_X)
    # -------------------------
    run_name = (
        f"{args.feature_name}"
        f"_txt{args.caption_name}"
        f"_{args.dataset}"
        f"_fold{args.fold}"
        f"_k{args.n_clusters}"
    )


    # Use fold splits
    split_train = f"{args.split_root}/fold{args.fold}/train.txt"
    split_val   = f"{args.split_root}/fold{args.fold}/val.txt"
    split_test  = f"{args.split_root}/fold{args.fold}/test.txt"

    # IMPORTANT: pass visual_dir/text_dir into dataset (requires VideoDataset supports these kwargs)
    data_val = VideoDataset(
        args.base_path,
        args.dataset,
        args.n_frames,
        standardise=args.std_feats,
        random=False,
        action_class=args.activity,
        split=split_val,
        visual_dir=args.visual_dir,
        text_dir=args.text_dir,
    )
    data_train = VideoDataset(
        args.base_path,
        args.dataset,
        args.n_frames,
        standardise=args.std_feats,
        random=True,
        action_class=args.activity,
        split=split_train,
        visual_dir=args.visual_dir,
        text_dir=args.text_dir,
    )
    data_test = VideoDataset(
        args.base_path,
        args.dataset,
        None,
        standardise=args.std_feats,
        random=False,
        action_class=args.activity,
        split=split_test,
        visual_dir=args.visual_dir,
        text_dir=args.text_dir,
    )

    val_loader   = DataLoader(data_val, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(data_test, batch_size=1, shuffle=False)

    if args.ckpt is not None:
        ssl = VideoSSL.load_from_checkpoint(args.ckpt)
    else:
        ssl = VideoSSL(
            layer_sizes=args.layers,
            layer_sizes_txt=args.layers_txt,
            n_clusters=args.n_clusters,
            beta_mm=args.beta_mm,
            use_mm_cost=args.use_mm_cost,
            alpha_train=args.alpha_train,
            alpha_eval=args.alpha_eval,
            ub_frames=args.ub_frames,
            ub_actions=args.ub_actions,
            lambda_frames_train=args.lambda_frames_train,
            lambda_frames_eval=args.lambda_frames_eval,
            lambda_actions_train=args.lambda_actions_train,
            lambda_actions_eval=args.lambda_actions_eval,
            step_size=args.step_size,
            train_eps=args.eps_train,
            eval_eps=args.eps_eval,
            radius_gw=args.radius_gw,
            n_ot_train=args.n_ot_train,
            n_ot_eval=args.n_ot_eval,
            n_frames=args.n_frames,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            rho=args.rho,
            exclude_cls=args.exclude,
            visualize=args.visualize,
        )

    # Optional wandb
    logger = (
        pl.loggers.WandbLogger(
            name=run_name,
            project="video_ssl",
            save_dir="wandb",
            group=args.group,
        )
        if args.wandb else None
    )

    trainer = pl.Trainer(
        devices=[args.gpu],
        check_val_every_n_epoch=args.val_freq,
        max_epochs=args.n_epochs,
        log_every_n_steps=50,
        logger=logger,
        default_root_dir="runs",
    )

    if args.k_means and args.ckpt is None:
        ssl.fit_clusters(train_loader, args.n_clusters)

    if not args.eval:
        trainer.validate(ssl, val_loader)
        trainer.fit(ssl, train_loader, val_loader)

    trainer.test(ssl, dataloaders=test_loader)

    # write run metadata inside version_X (after run)
    log_dir = trainer.log_dir
    if log_dir:
        log_dir = Path(log_dir)
        (log_dir / "run_name.txt").write_text(run_name + "\n")
        (log_dir / "run_args.txt").write_text(" ".join(os.sys.argv) + "\n")
