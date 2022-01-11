# coding: utf-8

import logging
import os
from shutil import ExecError

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler

from nebis.utils.schedule import get_linear_schedule_with_warmup
from nebis.utils import empty_hook, move_batch_to_device
from nebis.utils.evaluate import get_evaluator


class Base(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def profiled_fit(self, profile_dir, *args, **kwargs):
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=15, warmup=5, active=10, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
        ) as prof:

            _hook_step = kwargs["hook_step"]

            def hook_step(*args, **kwargs):
                _hook_step(*args, **kwargs)
                prof.step()

            kwargs["hook_step"] = hook_step
            self.fit(*args, **kwargs)

    def fit(
        self,
        dataset_train,
        dataset_test,
        config,
        hook_step=empty_hook,
        hook_epoch=empty_hook,
    ):
        optimizer = optim.Adam(
            self.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=config.weight_decay,
            amsgrad=False,
        )

        dataloader_train, sample_weight = dataset_train

        t_steps = len(dataloader_train) * config.epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(t_steps * config.warmup_percent),
            num_training_steps=t_steps,
        )

        # Starts training phase
        _i_step = 0
        for epoch in tqdm(range(config.epochs), position=0, leave=True):
            self.train()
            predictions = []
            targets = []
            embeddings = []
            losses = []

            pbar = tqdm(dataloader_train, position=0, leave=False)
            optimizer.zero_grad()
            for batch in pbar:
                batch = move_batch_to_device(batch, self.config.device)
                inputs = {"X_mutome": batch[0], "X_omics": batch[1]}
                target = batch[2]

                Y, H = self.forward(**inputs)

                loss = self.loss(Y, target, weight=sample_weight,)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                # Save step status
                losses.append(loss.item())

                Y_pred = self.Downstream.prediction(Y.detach())
                predictions += list(Y_pred)

                if torch.is_tensor(batch[2]):
                    target = target.detach().cpu().numpy()
                else:
                    target = [t.detach().cpu().numpy() for t in target]

                targets += list(target)

                embeddings.append(H.detach().cpu().numpy())

                hook_step(
                    epoch * len(dataloader_train) + _i_step,
                    {"loss": loss.item(), "learning rate": scheduler.get_last_lr()[0]},
                )
                _i_step += 1

            logging.info(
                "Epoch={},avg-CE-Loss={}".format(epoch, np.array(losses).mean())
            )
            hook_epoch(epoch)

            if epoch % self.config.checkpoint_interval == 0:
                logging.debug("Saving model checkpoint at Epoch {}".format(epoch))
                model_path = os.path.join(
                    self.config.model_out, "model_checkpoint_epoch_{}.pth".format(epoch)
                )
                self.save(model_path)

                logging.debug("Evaluating model at Epoch {}".format(epoch))
                Ys, Ps, Hs = self.predict(dataset_test)
                evaluator = get_evaluator(self.config.downstream)(Ys, Ps)
                evaluator.evaluate()

    def save(self, f):
        try:
            model_to_save = self.module if hasattr(self, "module") else self
            torch.save(
                model_to_save, f,
            )
        except Exception as e:
            print(e)
            print("Could not save the model at the specified location")

    def predict(self, dataset_test, hook=None):
        self.eval()
        Ys = []
        Ps = []
        Hs = []

        dataloader_test = dataset_test

        for batch in tqdm(dataloader_test, position=0, leave=True):
            batch = move_batch_to_device(batch, self.config.device)
            inputs = {"X_mutome": batch[0], "X_omics": batch[1]}
            target = batch[2]

            with torch.no_grad():
                Y, H = self.forward(**inputs)

            Y_pred = self.Downstream.prediction(Y.detach())
            if torch.is_tensor(batch[2]):
                target = target.detach().cpu().numpy()
            else:
                target = [t.detach().cpu().numpy() for t in target]

            Ps.append([Y, Y_pred])
            Ys.append(target)
            Hs.append(H.detach().cpu().numpy())

        Hs = np.concatenate(Hs).reshape(len(dataloader_test.dataset), -1)

        return Ys, Ps, Hs


def profiled_parallel_fit(profile_dir, *args, **kwargs):
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=15, warmup=5, active=10, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    ) as prof:

        _hook_step = kwargs["hook_step"]

        def hook_step(*args, **kwargs):
            _hook_step(*args, **kwargs)
            prof.step()

        kwargs["hook_step"] = hook_step
        parallel_fit(*args, **kwargs)


def parallel_predict(model, dataset_test, hook=None):
    model.eval()
    Ys = []
    Ps = []
    Hs = []

    dataloader_test = dataset_test

    for batch in tqdm(dataloader_test, position=0, leave=True):
        batch = move_batch_to_device(batch, model.module.config.device)
        inputs = {"X_mutome": batch[0], "X_omics": batch[1]}

        with torch.no_grad():
            Y, H = model.forward(**inputs)

        Y_pred = model.module.Downstream.prediction(Y.detach())
        if torch.is_tensor(batch[2]):
            target = target.detach().cpu().numpy()
        else:
            target = [t.detach().cpu().numpy() for t in target]

        Ps.append([Y, Y_pred])
        Ys.append(target)
        Hs.append(H.detach().cpu().numpy())

    Hs = np.concatenate(Hs).reshape(len(dataloader_test.dataset), -1)

    return Ys, Ps, Hs


def parallel_fit(
    model,
    dataset_train,
    dataset_test,
    config,
    hook_step=empty_hook,
    hook_epoch=empty_hook,
):
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=config.weight_decay,
        amsgrad=False,
    )

    dataloader_train, sample_weight = dataset_train

    t_steps = len(dataloader_train) * config.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(t_steps * config.warmup_percent),
        num_training_steps=t_steps,
    )

    # Starts training phase
    _i_step = 0
    for epoch in tqdm(range(config.epochs), position=0, leave=True):
        model.train()
        predictions = []
        targets = []
        embeddings = []
        losses = []

        pbar = tqdm(dataloader_train, position=0, leave=False)
        optimizer.zero_grad()
        for batch in pbar:
            batch = move_batch_to_device(batch, model.module.config.device)
            inputs = {"X_mutome": batch[0], "X_omics": batch[1]}
            target = batch[2]

            Y, H = model.forward(**inputs)

            loss = model.module.loss(Y, target, weight=sample_weight,)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Save step status
            losses.append(loss.item())

            Y_pred = model.module.Downstream.prediction(Y.detach())
            predictions += list(Y_pred)

            if torch.is_tensor(batch[2]):
                target = target.detach().cpu().numpy()
            else:
                target = [t.detach().cpu().numpy() for t in target]
            targets += list(target)

            embeddings.append(H.detach().cpu().numpy())

            hook_step(
                epoch * len(dataloader_train) + _i_step,
                {"loss": loss.item(), "learning rate": scheduler.get_last_lr()[0]},
            )
            _i_step += 1

        logging.info("Epoch={},avg-CE-Loss={}".format(epoch, np.array(losses).mean()))
        hook_epoch(epoch)

        if epoch % config.checkpoint_interval == 0:
            logging.debug("Saving model checkpoint at Epoch {}".format(epoch))
            model_path = os.path.join(
                config.model_out, "model_checkpoint_epoch_{}.pth".format(epoch)
            )
            model.module.save(model_path)

            logging.debug("Evaluating model at Epoch {}".format(epoch))
            Ys, Ps, Hs = parallel_predict(model, dataset_test)
            evaluator = get_evaluator(model.module.config.downstream)(Ys, Ps)
            evaluator.evaluate()
