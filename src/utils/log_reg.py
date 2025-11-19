
import logging
import random
import time
from collections import defaultdict
from math import log10
from warnings import simplefilter
import numpy as np
import torch
import torch.utils.data
from sklearn.exceptions import ConvergenceWarning
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import accuracy_score
from torchmetrics.classification import MulticlassAccuracy, MultilabelAccuracy, F1Score, AUROC
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from typing import List, Optional



class LogisticRegression:
    def __init__(self, C, max_iter, verbose, random_state=None, **kwargs):
        self.C = C
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.max_iter = max_iter
        self.random_state = random_state
        self.logreg = None
        self.verbose = verbose

    def compute_loss(self, feats, labels):
        loss = self.loss_func(feats, labels)
        wreg = 0.5 * self.logreg.weight.norm(p=2)
        return loss.mean() + (1.0 / self.C) * wreg

    def predict_proba(self, feats):
        assert self.logreg is not None, "Need to fit first before predicting probs"
        return self.logreg(feats).softmax(dim=-1)

    def predict(self, feats):
        assert self.logreg is not None, "Need to fit first before predicting classes"
        return self.predict_proba(feats).argmax(dim=-1)

    def fit(self, feats, labels):
        feat_dim = feats.shape[1]
        num_classes = len(torch.unique(labels))

        # set random seed
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self.logreg = torch.nn.Linear(feat_dim, num_classes, bias=True)
        self.logreg.weight.data.fill_(0.0)
        self.logreg.bias.data.fill_(0.0)

        # move everything to CUDA .. otherwise why are we even doing this?!
        self.logreg = self.logreg.to(feats.device)

        # define the optimizer
        opt = torch.optim.LBFGS(
            self.logreg.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=self.max_iter,
        )
        if self.verbose:
            pred = self.logreg(feats)
            loss = self.compute_loss(pred, labels)
            print(f"(Before Training) Loss: {loss:.3f}")

        def loss_closure():
            opt.zero_grad()
            pred = self.logreg(feats)
            loss = self.compute_loss(pred, labels)
            loss.backward()
            return loss

        opt.step(loss_closure)  # get loss, use to update wts

        if self.verbose:
            pred = self.logreg(feats)
            loss = self.compute_loss(pred, labels)
            print(f"(After Training) Loss: {loss:.3f}")

def _fit_logreg(
    feats: torch.Tensor,
    labels: torch.Tensor,
    cost: float,
    verbose: bool = False,
    max_iter: int = 100,
    use_sklearn: bool = False,
    multilabel : bool = False
) -> LogisticRegression:
    """
    Initialize and fit a `LogisticRegression` classifier for input features and
    labels. Default settings follow CLIP (L-BFGS, 1K iterations, etc.).
    """
    if use_sklearn:
        classifier = sk_LogisticRegression(
            C=cost, max_iter=max_iter, verbose=verbose
        )
        if multilabel:
            classifier = MultiOutputClassifier(classifier)
        feats = feats.cpu().numpy()
        labels = labels.cpu().numpy()
    else:
        classifier = LogisticRegression(
            C=cost, max_iter=max_iter, verbose=verbose
        )
        if multilabel:
            raise NotImplementedError()
    classifier.fit(feats, labels)
    return classifier

def train_linear_probe(
    train_feats,
    train_labels,
    valid_feats,
    valid_labels,
    use_mean_accuracy=True,
    sk_verbose=False,
    max_iter=100,
    combine_trainval=True,
    use_sklearn=False,
    multilabel=False,
    fastsearch=False,
    logger = None
):
    
    device = train_feats.device
    average = "macro" if use_mean_accuracy else "micro"
    
    NUM_C = len(train_labels.unique())
    acc_meter = MulticlassAccuracy(
        num_classes=NUM_C,
        average=average).to(device)

    # CLIP performed linear probe evaluation by sweeping over 96 log-spaced costs.
    # Following CLIP, we sweep in two stages (coarse and fine) for quick search.
    costs = [1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6]
    logger.debug(f"First sweep with costs: {costs}")

    # Train and avaluate each classifier and get accuracy.
    accuracies = []
    for cost in costs:
        classifier = _fit_logreg(
            train_feats, train_labels, cost, sk_verbose, max_iter, use_sklearn, multilabel
        )
        if use_sklearn:
            predictions = torch.from_numpy(classifier.predict(valid_feats.cpu().numpy())).to(device)
        else:
            predictions = classifier.predict_proba(valid_feats)
        accuracy = acc_meter(predictions, valid_labels)
        accuracies.append(accuracy)

        acc_meter.reset()
        logger.debug(f"Cost = {cost}, Top-1 accuracy = {accuracy:.3f}")

    best_accuracy = max(accuracies)
    best_cost = costs[accuracies.index(best_accuracy)]

    if not fastsearch:
        # Second sweep: search around the best cost with a resolution of 8 steps per
        # decade. Example: if best cost = 1e2, then these costs will be in (1, 1e-4).
        costs = torch.logspace(log10(best_cost) - 2, log10(best_cost) + 2, 29)
        costs = costs[(costs >= 1e-6) & (costs <= 1e6)].tolist()

        # We may visit the same cost value multiple times while searching, to avoid
        # re-training the classifier, keep a map of accuracies per cost.
        accuracies = {best_cost: best_accuracy}

        logger.debug("Performing second sweep as a binary search around best cost.")
        logger.debug(f"Initial search space: {[round(c, 3) for c in costs]}")

        while len(costs) > 1:
            # Get mid-points of left/right half interval of search space: (25,50,75)%
            cost_25 = costs[len(costs) // 4]
            cost_50 = costs[len(costs) // 2]
            cost_75 = costs[-len(costs) // 4]
            logger.debug(
                f"Half interval mid-points: {cost_25=:.3f}, {cost_50=:.3f}, {cost_75=:.3f}"
            )

            # Compute accuracy for these costs (skip if computed in prev iteration).
            for cost in [cost_25, cost_50, cost_75]:
                _acc = accuracies.get(cost, None)
                if _acc is None:
                    classifier = _fit_logreg(
                        train_feats, train_labels, cost, sk_verbose, max_iter, use_sklearn, multilabel
                    )
                    if use_sklearn:
                        predictions = classifier.predict(valid_feats.cpu().numpy())
                        _acc = acc_meter(torch.from_numpy(predictions).to(device), valid_labels)
                    else:
                        predictions = classifier.predict_proba(valid_feats)
                        _acc = acc_meter(predictions, valid_labels)
                    accuracies[cost] = _acc
                    acc_meter.reset()

                logger.debug(f"Cost = {round(cost, 3)}, Top-1 accuracy = {_acc:.3f}")

            # Cut down the search space by half such that the mid-point of the resulting
            # reduced search space is the cost with current best accuracy.
            max_acc = max(accuracies[cost_25], accuracies[cost_50], accuracies[cost_75])
            costs = (
                costs[: len(costs) // 2]
                if max_acc == accuracies[cost_25]
                else costs[len(costs) // 2 :]
                if max_acc == accuracies[cost_75]
                else costs[len(costs) // 4 : -len(costs) // 4]
            )
            logger.debug(f"Reduced search space, costs: {[round(c, 3) for c in costs]}")

        # Filter None accuracy values (some costs may not be visited while searching).
        # Then find best accuracy and its cost.
        best_cost, best_accuracy = max(accuracies.items(), key=lambda k: k[1])

    logger.debug(f"Best cost = {best_cost:.3f}, Top-1 accuracy = {best_accuracy:.3f}")

    # train final classifier
    if combine_trainval:
        trainval_feats = torch.cat([train_feats, valid_feats], dim=0)
        trainval_labels = torch.cat([train_labels, valid_labels], dim=0)

        final_classifier = _fit_logreg(
            trainval_feats,
            trainval_labels,
            best_cost,
            sk_verbose,
            max_iter,
            use_sklearn,
            multilabel
        )
    else:
        final_classifier = _fit_logreg(
            train_feats, train_labels, best_cost, sk_verbose, max_iter, use_sklearn, multilabel
        )

    return final_classifier

def evaluate_linear_probe(
    train_feats,
    train_labels,
    test_feats,
    test_labels,
    val_feats=None,
    val_labels=None,
    multilabel=False,
    holdout_fraction=0.6,
    use_mean_accuracy=True,
    sk_verbose=False,
    max_iter=100,
    combine_trainval=True,
    use_sklearn=False,
    fastsearch=False,
    logging_level="INFO"
):
    """
    Args:
        holdout_fraction: Fraction of the (official) train split to hold out for
            validation while searching the cost hyperparameter. Holding out will
            be deterministic and have similar class distribution as train split.
    """
    start = time.time()
    logger = logging.getLogger("Linear probing")
    logger.setLevel(logging_level)
    if val_feats is None or val_labels is None:
        if multilabel: # simplify the problem
            train_idx, val_idx = next(ShuffleSplit(train_size=holdout_fraction, random_state=48).
                                      split(np.ones((len(train_feats), 1))))
        else:
            train_idx, val_idx = next(StratifiedShuffleSplit(train_size=holdout_fraction, random_state=48).
                                      split(np.ones((len(train_feats), 1)), train_labels.cpu().numpy()))
        val_feats = train_feats[val_idx]
        val_labels = train_labels[val_idx]
        train_feats = train_feats[train_idx]
        train_labels = train_labels[train_idx]

    classifier = train_linear_probe(
        train_feats,
        train_labels,
        val_feats,
        val_labels,
        use_mean_accuracy,
        sk_verbose,
        max_iter=max_iter,
        combine_trainval=combine_trainval,
        use_sklearn=use_sklearn,
        multilabel=multilabel,
        fastsearch=fastsearch,
        logger=logger
    )
    test_acc = test_linear_probe(classifier, test_feats, test_labels, use_mean_accuracy,
                                 use_sklearn=use_sklearn, multilabel=multilabel, logger=logger)

    del classifier
    torch.cuda.empty_cache()
    logger.debug(f"Time taken {time.time() - start:.2f}")
    return test_acc['acc1']


def test_linear_probe(
    linear_classifier, test_feats, test_labels, use_mean_accuracy,
        use_sklearn=False, num_classes=None, multilabel=False, logger=None
):
    if logger is None:
        logger = logging.getLogger("Linear probing")
    # evaluate
    average = "macro" if use_mean_accuracy else "micro"
    device = test_feats.device
    if multilabel:
        num_labels = test_labels.shape[-1]
        acc1 = MultilabelAccuracy(num_labels=num_labels, average=average).to(device)
        f1_score = F1Score(task="multilabel", num_labels=num_labels, average=average).to(device)
        f1_weighted_score = F1Score(task="multilabel", num_labels=num_labels, average="weighted").to(device)
        if use_sklearn:
            predictions = torch.as_tensor(linear_classifier.predict(test_feats.cpu().numpy())).to(device)
        else:
            predictions = linear_classifier.predict(test_feats)
        accuracy1 = float(acc1(predictions, test_labels))
        f1_mean = float(f1_score(predictions, test_labels))
        f1_weighted = float(f1_weighted_score(predictions, test_labels))
        acc1_sklearn = accuracy_score(test_labels.cpu().numpy(), predictions.cpu().numpy())
        logger.info(f"Test acc@1/acc@1(subset)/f1-score/f1-weighted: {accuracy1:.3f}/{acc1_sklearn:.3f}/{f1_mean:.3f}/{f1_weighted:.3f}")
        return {"acc1": accuracy1, "f1_mean": f1_mean, "f1_weighted": f1_weighted, "acc1(subset)": acc1_sklearn}
    else:
        NUM_C = len(set(test_labels.cpu().numpy())) if num_classes is None else num_classes
        acc1 = MulticlassAccuracy(num_classes=NUM_C, average=average).to(test_feats.device)
        acc_per_class = MulticlassAccuracy(num_classes=NUM_C, average=None).to(test_feats.device)
        acc5 = MulticlassAccuracy(num_classes=NUM_C, average=average,
                                  top_k=min(NUM_C, 5)).to(test_feats.device)
        roc_auc = AUROC(task="multiclass", num_classes=NUM_C).to(test_feats.device)
        if use_sklearn:
            predictions = torch.as_tensor(linear_classifier.predict_proba(test_feats.cpu().numpy())).to(device)
        else:
            predictions = linear_classifier.predict_proba(test_feats)
        accuracy1 = float(acc1(torch.as_tensor(predictions), test_labels))
        accuracy5 = float(acc5(torch.as_tensor(predictions), test_labels))
        accuracy_per_class = [float(x) for x in acc_per_class(torch.as_tensor(predictions), test_labels)]
        auc = float(roc_auc(torch.as_tensor(predictions), test_labels))
        logger.info(f"Test acc@1/acc@5/acc_per_class/roc_auc: {accuracy1:.3f}/{accuracy5:.3f}/{accuracy_per_class}/{auc}")
        return {"acc1": accuracy1, "acc5": accuracy5, "roc_auc": auc}
