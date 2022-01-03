# coding: utf-8

import logging
from shutil import ExecError

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler

from bideset.utils.schedule import get_linear_schedule_with_warmup


class Base(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def profiled_fit(self, dataloader, config, *args, **kwargs):
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=15, warmup=5, active=10, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.profile_dir),
        ) as prof:
            def hook_step():
                prof.step()

            self.fit(*args, dataloader, config, hook_step=hook_step, *args, **kwargs)

    def fit(self, dataset_train, dataset_test, config, hook_step=None, hook_epoch=None):
        optimizer = optim.Adam(
            self.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=config.weight_decay,
            amsgrad=False,
        )

        dataloader_train, sample_weight = dataset_train.fitting()

        t_steps = len(dataloader_train) * config.epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(t_steps * config.warmup), num_training_steps=t_steps,
        )
        
        # Starts training phase
        _i_step = 0
        for epoch in tqdm(range(config.epochs), position=0, leave=True):
            self.train()
            predictions = []
            labels = []
            embeddings = []
            losses = []

            pbar = tqdm(dataloader_train, position=0, leave=False)
            optimizer.zero_grad()
            for batch in pbar:
                batch = tuple(t.to(self.config.device) for t in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                }
                Y, H = self.forward(**inputs)

                loss = self.loss(
                    Y,
                    H,
                    weight=sample_weight,
                )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                Y_pred = self.Downstream.prediction()

                predictions += list(Y_pred.detach().cpu().numpy())
                labels += list(batch[3].detach().cpu().numpy())
                embeddings.append(H.detach().cpu().numpy())
                losses.append(loss.item())

                try:
                    self.writer.add_scalar(
                        "training loss", loss.item(), epoch * len(dataloader_train) + _i_step
                    )
                    self.writer.add_scalar(
                        "learning rate",
                        scheduler.get_lr()[0],
                        epoch * len(dataloader_train) + _i_step,
                    )
                except:
                    logging.error("#Could not register the loss")

                hook_step()
                _i_step += 1

            logging.info(
                "Epoch={},avg-CE-Loss={}".format(epoch, np.array(losses).mean())
            )
            hook_epoch()

            if epoch % self.config.chechpoint_save_interval == 0:
                logging.debug("Saving model checkpoint at Epoch {}".format(epoch))
                self.save()

                logging.debug("Evaluating model at Epoch {}".format(epoch))
                self.predict(dataset_test)
                

    def save(self, f):
        try:
            model_to_save = self.module if hasattr(self, "module") else self
            torch.save(
                model_to_save,
                f,
            )
        except:
            raise ExecError("Could not save the model at the specified location")
        
    def predict(self, dataset, hook=None):
        self.eval()
        Ys = []
        Ps = []
        Hs = []

        dataloader = dataset.prediction()

        for batch in tqdm(dataloader, position=0, leave=True):
            batch = tuple(t.to(self.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
            }

            with torch.no_grad():
                Y, H = self.forward(**inputs)

            Y_pred = self.Downstream.prediction()

            Ps += list(Y_pred.detach().cpu().numpy())
            Ys += list(Y.detach().cpu().numpy())
            Hs.append(H.detach().cpu().numpy())

        Hs = np.concatenate(np.array(Hs)).reshape(len(Ps), -1)

        return Ys, Ps, Hs


class Configurator():
    def __init__(self, config):
        super().__init__()
        self.config = config