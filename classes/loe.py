"""
# Copyright 2018 Professorship Media Informatics, University of Applied Sciences Mittweida
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @author Richard Vogel, 
# @email: richard.vogel@hs-mittweida.de
# @created: 06.10.2019
"""
import numpy as np
import numpy.random
from numpy.random import RandomState
from abc import abstractmethod
from sklearn.utils.validation import check_is_fitted, check_random_state, check_X_y
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Optional, List, Union, Dict, Callable, Tuple, Type
import logging
import datetime
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from joblib import parallel, delayed, Parallel
from numpy.random import Generator, PCG64


def _parallel_fit(model_cls: Type[BaseEstimator],
                  X: np.ndarray,
                  y: np.ndarray,
                  model_params: Dict[str, any],
                  model_idx: int = 0) -> Tuple[BaseEstimator, int]:
    """
    This function is a fit-wrapper
    be aware that it will create NEW model instances when called (you have to replace the old variants)
    this is due to avoid copying expensive structures between processes to limit the run time to a minumum

    :param model_cls:
    :param X:
    :param y:
    :param model_params:
    :param model_idx: will be returned as-is in order to match instances back
    :return:
    """
    model = model_cls(**model_params)
    model.fit(X, y)

    return model, model_idx


class CustomDSBase(BaseEstimator, ClassifierMixin):
    """
    Base Model for classifiers that build their during training process
    """

    # if no classifier is given that is the amount of
    # standard created classifiers
    _default_pool_size: int = 10

    def __init__(self,
                 pool_classifiers: Optional[Union[List[BaseEstimator], BaseEstimator]] = None,
                 val_perc: Optional[float] = 0.0,
                 DSEL_perc: Optional[float] = None,
                 random_state: Optional[Union[int, RandomState]] = None,
                 needs_proba: bool = False, max_jobs: int=1):

        self._val_perc: Optional[float] = val_perc
        self.random_state = random_state
        self.pool_classifiers = pool_classifiers
        self.needs_proba = needs_proba
        self.DSEL_perc = DSEL_perc
        self.pool_classifiers_: Optional[List[BaseEstimator]] = pool_classifiers
        self.random_state_: Optional[RandomState] = None
        self.random_generator_: Optional[Generator] = None
        self.positions_trace_: Optional[List[int]] = None

        self.n_classifiers_: Optional[int] = None
        self.max_jobs = max_jobs

        self.classes_: Optional[List[str]] = None
        self.enc_: Optional[LabelEncoder] = None
        self.n_classes_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.n_samples_: Optional[int] = None
        self._is_fitted: bool = False

        self.train_target_: Optional[np.ndarray] = None
        self.train_data_: Optional[np.ndarray] = None
        self.train_data_processed_: Optional[np.ndarray] = None
        self.train_data_processed_target_: Optional[np.ndarray] = None

        self.val_target_: Optional[np.ndarray] = None
        self.val_data_: Optional[np.ndarray] = None
        self.val_data_processed_: Optional[np.ndarray] = None
        self.val_data_processed_target_: Optional[np.ndarray] = None

        self.DSEL_data_: Optional[np.ndarray] = None
        self.DSEL_processed_: Optional[np.ndarray] = None
        self.DSEL_target_: Optional[np.ndarray] = None
        self.DSEL_processed_target_: Optional[np.ndarray] = None

        self.X_expert: Optional[np.ndarray] = None
        self.X_expert_dsel: Optional[np.ndarray] = None
        self.X_expert_val: Optional[np.ndarray] = None

        self.scaler_: Optional[StandardScaler] = None

    def get_max_jobs(self) -> int:
        """
        Returns the number of jobs thats launched in parallel at maximum
        :return:
        """
        return self.max_jobs

    def set_max_jobs(self, n_jobs:int):
        """
        Sets the maximum amount of jobs to be run in parallel
        :param n_jobs:
        :return:
        """
        assert n_jobs >= 1, "There has to be at least one job to be run in parallel (Code: 894723894)"
        self.max_jobs = n_jobs

    def reset(self):
        """
        Will put the classifier into an unfit status
        by reseting all variables ending with _

        :return:
        """
        for key, value in self.__dict__.items():
            if key[-1] == '_':
                if len(key) > 1 and key[:-2] != '__':
                    self.__setattr__(key, None)

        self._is_fitted = False

    def score(self, X, y, X_expert=None, sample_weight=None):
        """
        Will calculate the accuracy of the given data
        :param X:
        :param y:
        :param X_expert:
        :param sample_weight:
        :return:
        """
        if sample_weight is not None:
            raise NotImplementedError("Sorry, but weighted score is not supported atm (Code: 382472389)")

        # y = self._preprocess_y(y)
        score = accuracy_score(y, self.predict(X, X_expert))
        return score

    def estimate_competence(self,
                            query):
        """
        Parameters
        ----------
        query : array of shape  = [n_samples, n_features]
            The test examples.


        Returns
        -------
        competences : array of shape = [n_samples, n_classifiers]
            Competence level estimated for each base classifier and test
            example.
        """
        assert self.is_fitted(), "The model has to be fitted to " \
                                 "estimate competence (Code: 2384023)"

    @abstractmethod
    def select(self, query: np.ndarray):
        """
        Should select the base classifier given a query
        :return:
        """
        pass

    def fit(self, X, y, X_expert=None):
        """
        Fit the classifier to the given data.

        Parameters
        ----------
        :param X: Training AND DSEL data (if DSEL_perc is None or 1. thats the same)
        :param X_expert: If provided that data is handed of to the experts instead of the LoE data. May be used if your expert can handle specific data that LoE cannot (e.g. Categorical, Missing Vals etc.)
        :param y: Class labels

        :raises ValueError: if classifiers are not supported
        :return:
        """

        self.reset()
        self.random_generator_ = np.random.default_rng(seed=self.random_state)
        self.random_state_ = np.random.RandomState(seed=self.random_state)
        # Check if the length of X and y are consistent.
        X, y = check_X_y(X, y)

        if X_expert is not None:
            assert len(X_expert) == len(X), "If you provide expert data, " \
                                            "that data has to match the normal training data! (Code: 932874923)"
        # Check if the pool of classifiers is `None` or a BaseEstimator.
        # If yes, use a BaggingClassifier for the pool.
        if not isinstance(self.pool_classifiers, List):
            if len(X) < 2:
                raise ValueError('More than one sample is needed '
                                 'if the pool of classifiers is not informed.')

            if not isinstance(self.pool_classifiers, BaseEstimator):
                self.pool_classifiers = None

            self.pool_classifiers_ = [DecisionTreeClassifier(random_state=self.random_state_,
                                                             max_depth=4)
                                      for i in range(self._default_pool_size)]

        else:
            self.pool_classifiers_ = self.pool_classifiers

        if not self._check_base_classifiers_prefit():
            raise ValueError("At least one of your models is not a classifier "
                             "or needs a predict_proba method (Code: 8738495635464719)")

        # split validation data
        X_val = y_val = X_expert_val = None
        y_labels = set(y)
        if self._val_perc is not None and self._val_perc > 0.:
            indices_val = np.arange(len(X))
            X_train, X_val, y_train, y_val, X_train_indices, y_val_indices = train_test_split(X, y, indices_val,
                                                                                              test_size=self._val_perc,
                                                                                              random_state=self.random_state_)
            if len(set(y_val)) != len(y_labels) or len(set(y)) != len(y_labels):
                logging.getLogger().warning("When splitting the data into training and validation data, "
                                            "at least one of the set is too small to contain at least one data "
                                            "point of each label. I will not build a validation set! (Code: 9482903964)")
                X_val = y_val = None
            else:
                X = X_train
                y = y_train
                X_expert_val = X_expert[y_val_indices] if X_expert is not None else None
                X_expert = X_expert[X_train_indices] if X_expert is not None else None

        # if we want to split the train and DSEL set
        if self.DSEL_perc is None or self.DSEL_perc == 0.:  # use all data as DSEL
            X_dsel = X.__copy__()
            y_dsel = y.__copy__()

            if X_expert is not None:
                X_expert_dsel = X_expert.__copy__()
            else:
                X_expert_dsel = None
        else:
            # generate split
            indices = np.arange(len(X))
            X, X_dsel, y, y_dsel, X_indices, X_dsel_indices = train_test_split(
                X, y, indices, test_size=self.DSEL_perc,
                random_state=self.random_state_, stratify=y)

            if X_expert is not None:
                #X_expert, X_expert_dsel, y_expert_dsel, y_expert_dsel = train_test_split(
                #    X, y, test_size=self.DSEL_perc,
                #    random_state=state_num, stratify=y)
                X_expert_dsel = X_expert[X_dsel_indices]
                X_expert = X_expert[X_indices]
            else:
                X_expert_dsel = None

        self.n_classifiers_ = len(self.pool_classifiers_)
        self._setup_label_encoder(y)
        self._set_train_data(X,
                             y,
                             X_expert=X_expert)

        self._set_dsel(X_dsel,
                       y_dsel,
                       X_expert=X_expert_dsel)

        self._set_val_data(X_val, y_val, X_expert_val)

        self.fit_ensemble()
        self._is_fitted = True

    @abstractmethod
    def fit_ensemble(self):
        """
        This method implements the main algorithm
        :return:
        """
        pass

    def _set_val_data(self, X, y, X_expert=None):
        """Sets the validation data for estimating generalizing properties of the model

           Parameters
           ----------
           X : array of shape = [n_samples, n_features]
               The Input data.

           y : array of shape = [n_samples]
               class labels of each sample in X.

           X_expert : If provided that data is handed over to experts instead of LoE-transformed data
           """

        self.val_data_ = X
        self.val_target_ = y

        if self.val_data_ is not None:
            self.val_data_processed_ = self._preprocess_X(X=X)
            self.val_data_processed_target_ = self._preprocess_y(y=y)

            self.val_data = X
            self.val_data_target_ = y

            if X_expert is not None:
                self.X_expert_val = X_expert
            else:
                self.X_expert_val = None

    def _set_train_data(self, X, y, X_expert=None):
        """Sets the train data of the model which will be used for fotting ensemble members

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The Input data.

        y : array of shape = [n_samples]
            class labels of each sample in X.

        X_expert : If provided that data is handed over to experts instead of LoE-transformed data
        """

        self.n_classes_ = self.classes_.shape[0]
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]

        self.train_data_ = X
        self.train_target_ = y
        self.train_data_processed_ = self._preprocess_X(X=X)
        self.train_data_processed_target_ = self._preprocess_y(y=y)

        self.train_data_ = X
        self.train_data_target_ = y

        if X_expert is not None:
            self.X_expert = X_expert
        else:
            self.X_expert = None

    def _set_dsel(self, X, y, X_expert=None):
        """
        Sets dsel data that will be used
        :param X:
        :param y:
        :param X_expert:
        :return:
        """

        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)

        self.dsel_std_ = X.std(axis=0) + 10e-5
        self.dsel_max = X.max(axis=0)
        self.dsel_min = X.min(axis=0)

        self.DSEL_data_ = X
        self.DSEL_target_ = y

        self.DSEL_processed_ = self._preprocess_dsel(X=X)
        self.DSEL_processed_target_ = self._preprocess_y(y=y)

        self.X_expert_dsel = X_expert

    def _preprocess_dsel(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocesses the data for use with DSEL

        :param X:
        :return:
        """
        return self.scaler_.transform(X)

    def _preprocess_X(self, X: np.ndarray):
        """
        Will preprocess the X (feature data)
        :return:
        """
        return X

    def _preprocess_y(self, y: np.ndarray):
        """
        Will preprocess labels

        :param y:
        :return:
        """
        return self.enc_.transform(y)

    def _check_base_classifiers_prefit(self) -> bool:
        """
        Check sanity of base classifiers BEFORE fit
        :return:
        """

        probabilistic_ok = not self.needs_proba or np.all([hasattr(clf, 'predict_proba')
                                                           for clf in self.pool_classifiers_])

        is_classifier = np.all(isinstance(clf, ClassifierMixin) for clf in self.pool_classifiers_)

        return probabilistic_ok and is_classifier

    def _check_base_classifier_fitted(self):
        """ Checks if each base classifier in the pool is fitted.

        Raises
        -------
        NotFittedError: If any of the base classifiers is not yet fitted.
        """
        for clf in self.pool_classifiers:
            check_is_fitted(clf, "classes_")

    def _setup_label_encoder(self, y):
        self.enc_ = LabelEncoder()
        self.enc_.fit(y)
        self.classes_ = self.enc_.classes_

    def is_fitted(self) -> bool:
        """
        Checks if the model is fit
        :return:
        """
        return self._is_fitted

    def log(self, level: int, msg: str):
        """
        Logs a message

        :param level:
        :param msg:
        :return:
        """
        # print(msg)
        logging.getLogger(self.__class__.__name__).log(level, msg)

    def log_info(self, msg):
        self.log(level=logging.INFO, msg=msg)

    def log_warning(self, msg):
        self.log(level=logging.WARNING, msg=msg)

    def log_debug(self, msg):
        self.log(level=logging.DEBUG, msg=msg)

    def predict(self, X: np.ndarray, *args, **kwargs) -> float:
        pass


class CustomDCSBase(CustomDSBase):
    pass


class LoE(CustomDCSBase):
    def __init__(self,
                 pool_classifiers: Optional[Union[List[BaseEstimator], BaseEstimator]] = None,
                 step_size: float = 2.,
                 iterations: int = 20,
                 maximum_selected_features: Optional[Union[int, str]] = 'auto',
                 fixed_classifiers: Optional[List[int]] = None,
                 assign_random_points_if_model_outside: bool = True,
                 stochastic_attention: float = 1.,
                 step_callback: Optional['BaseLoEStepCallBack'] = None,
                 mode: str = 'global',
                 *args,
                 **kwargs,
                 ):
        """
        The LoE model. For beginners the settings might mostly be let alone. However, there might be great potential
        in tweaking parameters of the model (start with `iterations` and `step_size`)

        Parameters
        ----------
        :param step size: determines the movement speed
        :param iterations: amount of cycles
        :param fixed_classifiers: List of len(pool_classifiers)
        :param maximum_selected_features: if not `None` and all classifiers support feature_importances only up to maximum_selected_features will be taken into account when calculating distances. 'auto' will include features until 60% of information for assigning is saturated
        :param assign_random_points_if_model_outside: force some random points to be assigned to the model even if its
        :param stochastic_attention: If < 1 will use only a subset of DSEL data to validate against each iteration
        outside current assignments area (Prevent dying models)
        where and entry of True means the classifier will NOT be refit during process

        References
        ----------
    """
        super().__init__(pool_classifiers=pool_classifiers,
                         *args,
                         **kwargs)

        # how many stdevs one step should walk within each feature dimension
        self._step_size = step_size

        # how many cycles one fit should do
        self._iterations = iterations

        # if True, if a model does not get a regular assignemt, we assign some random points to not let it die out
        self._assign_random_points_if_model_outside = assign_random_points_if_model_outside

        # stores model positions in feature space
        self.model_positions_: Optional[np.ndarray] = None

        # callback to intercept learning process
        self._step_callback = step_callback

        self._maximum_selected_features = maximum_selected_features

        # current values of feature importances
        self.current_feature_importances_ : Optional[np.ndarray] = None

        # will store the features that are important right now as index for faster access
        self.important_feature_indexer_: Optional[np.ndarray] = None

        assert self._iterations >= 1, "You have to iterate at least 1 round (Code: 237498237)"
        assert self._step_size > 0, "Step size has to be greater than zero (Code: 823742893)"
        assert mode in ('local', 'global', 'mixed'), "Mode has to be either local, global or mixed (Code: 9996573482)"
        assert 0. < stochastic_attention <= 1., f"Stochastic attention level " \
                                                                         f"has to be between " \
                                                                         f"]0, 1]. (Code: 93284209348)"

        # if local region of competence will be determinded only for current assignment
        self._mode = mode

        # if not None will use only use the given proportion of samples for each step (uniformly drawn)
        self._stochastic_attention = stochastic_attention

        # if classifiers are not marked fixed or not fixed we mark them as non-fixed (all of them)
        self._fixed_classifiers = fixed_classifiers if fixed_classifiers is not None \
            else [False for i in range(self._default_pool_size \
                                           if not self.pool_classifiers else len(self.pool_classifiers))]

        # Check some preconditions
        if self.pool_classifiers is not None:
            assert len(self._fixed_classifiers) == len(self.pool_classifiers), f"If you define fixed classifiers " \
                                                                               f"you have to provide that state for each " \
                                                                               f"classifier (#{len(self.pool_classifiers)}) " \
                                                                               "(Code: 2398420398)"
        if np.any(self._fixed_classifiers):
            if not self.pool_classifiers:
                raise ValueError(f"You have to provide fitted classifiers if you want to fix them (Code: 8789234563")
            for i, clf in enumerate(self.pool_classifiers):
                if self._fixed_classifiers[i] is True and not 'classes_' in dir(clf):
                    raise ValueError(f"Classifier index {i+1} is marked fixed but is not fitted (Code: 23742893)")

    def estimate_competence(self,
                            query):

        super().estimate_competence(query)

        X = np.atleast_2d(query)
        competences = np.zeros(shape=(len(query), len(self.get_current_classifiers())),
                               dtype=bool)

        assignments = self.assign_data_points_to_model(X, is_processed=False)

        for model_idx, assignment in enumerate(assignments):
            competences[assignment, model_idx] = 1

        return competences

    def _preprocess_dsel(self, X: np.ndarray) -> np.ndarray:
        return X

    def step(self, step_size: float):
        """
        Do one step (Pulling + fitting in that order)

        :param step_size:
        :return:
        """
        self._update_feature_importances()
        self._pull_classifiers_to_competence_area(step_size=step_size)
        self._fit_classifiers_to_closest_points()

    def _update_feature_importances(self, alpha: float = 0.3):
        """
        Will be performed before each explicit step
        to update the current feature importances

        Imporances will be alpha * current_importances + (1 - alpha) * last_importances
        :return:
        """
        assert 0 < alpha <= 1, "Mixing factor should be in range ]0...1] (Code: 239482093)"

        if self._maximum_selected_features is not None:
            max_features = self._maximum_selected_features
            last_importances = self.current_feature_importances_

            # if called first time we do not have feature importances yet
            # so we pretend equal distribution
            if last_importances is None:
                # we do not have last importances so we will equally distribute importances....
                last_importances = np.ndarray(shape=(self.get_data_dimensionality(), ), dtype=float)
                last_importances[:] = 1. / self.get_data_dimensionality()
                self.current_feature_importances_ = last_importances

                # ... and NOT lasso anything out first iteration
                self.important_feature_indexer_ = np.ones(shape=(self.get_data_dimensionality(),), dtype=bool)
            else:
                # we have some importances from last round and will lasso something out
                current_importances = self.feature_importances_

                # mix em
                self.current_feature_importances_ = alpha * current_importances + (1 - alpha) * last_importances

                # before expensive sorting we check if any feature selection would result in a change at all
                # since selecting more features than available is selecting all features
                if max_features >= len(self.current_feature_importances_):
                    # spill out a info and just leave here
                    self.log_info(f"Your data set consists of {len(self.current_feature_importances_)} features. "
                                  f"However, you try to select {max_features} features, which results in selecting "
                                  f"all features anyways. (Code: 39048230942)")
                    # noop
                else:
                    # sort feature importances (will be sorted lowest to highest per default)
                    sorted_importances = np.sort(self.current_feature_importances_)

                    # get cutoff value
                    threshold = sorted_importances[-max_features]

                    # and update indexer
                    self.important_feature_indexer_ = self.current_feature_importances_ >= threshold

    def fit_ensemble(self):
        start_time = datetime.datetime.now()

        # pre estimate selected features if the lasso is set to auto
        if self._maximum_selected_features == 'auto':
            self.auto_determine_maximum_selected_features()

        self.model_positions_ = self._init_model_positions()
        self.positions_trace_ = [self.model_positions_.copy()]
        self._fit_classifiers_to_closest_points()
        best_performance = self._calculate_performance()
        best_pos = self.model_positions_.copy()
        best_feature_importances = self.get_important_feature_indexer()
        self.log_info(f"Start performance: {best_performance:.3f}\n")
        if self._mode == 'mixed':
            mixed_mode = True
            self._mode = 'global'
        else:
            mixed_mode = False
        for i in range(self._iterations):
            if mixed_mode is True and i == self._iterations // 2:
                self.log_info("Switched from global model to local mode (Fine-tuning)")

            round_time = datetime.datetime.now()
            self.log_info(f"Performing round {i}")
            step_size = self._decay_fn(i, self._iterations) * self._step_size
            self.log_info(f"Current step size: {step_size:.3f}")

            self.step(step_size=step_size)
            self.positions_trace_.append(self.model_positions_.copy())

            current_performance = self._calculate_performance()
            self.log_info(f"Performance: {current_performance:.3f}")

            if current_performance > best_performance:
                self.log_info(f"Performance increase by {(current_performance-best_performance):.3f}")
                best_performance = current_performance
                best_pos = self.model_positions_.__copy__()
                best_feature_importances = self.get_important_feature_indexer()

            time_delta = datetime.datetime.now() - round_time

            if self._step_callback is not None:
                self._step_callback(iteration=i,
                                    model=self,
                                    performance_train=current_performance)

            self.log_info(f"Round took {time_delta.total_seconds():.3f} seconds\n")

            # if current_performance == 1.:
            #    self.log_info(f"Stopping due to performance is 1")
            #    break

        # restore best model positions
        self.model_positions_ = best_pos
        self.important_feature_indexer_ = best_feature_importances
        self._fit_classifiers_to_closest_points()

        # recalculate performance
        performance = self._calculate_performance()
        train_delta = datetime.datetime.now() - start_time
        self.log_info(f"Performance Final: {performance:.3f}")
        self.log_info(f"Training took {train_delta.total_seconds():.3f} seconds in sum")

    def _decay_fn(self, current_iteration: int, max_iteration: int) -> float:
        """
        Will produce a factor that can be used as decay for iteratively
        reduced parameters

        :param current_iteration:
        :param max_iteration:
        :return:
        """
        return (max_iteration - current_iteration)/(max_iteration)

    def _pull_classifiers_to_competence_area(self, step_size: float = 0.8):
        """
        Will use DSEL to pull models to working clusters
        worky differently depending on current mode (global, local)
        :return:
        """
        models = self.get_current_classifiers()
        dsel_data = self.get_DSEL(processed=True)
        dsel_labels = self.get_DSEL_target(processed=True)
        model_positions = self.model_positions_

        # only take some points into account if requested (stochastic attention)
        if self._stochastic_attention < 1.:
            # we leave the "draw with replacement"-option turned on
            # to reduce calculation time here
            data_indices = self.random_generator_.choice(np.arange(0, len(dsel_data), 1),
                                                     int(self._stochastic_attention * len(dsel_data))
                                            )

            data_to_evaluate = dsel_data[data_indices]
            dsel_labels = dsel_labels[data_indices]
        else:
            # otherwise we use whole DSEL
            data_to_evaluate = dsel_data

        # local mode only predicts the already assigned neighbourhood
        if self._mode == 'local':
            assignments = self.assign_data_points_to_model(dsel_data, is_processed=True)
        else:
            assignments = np.arange(0, len(data_to_evaluate), 1)

        for i, model in enumerate(models):
            pos = model_positions[i]

            if self._mode == 'local':
                assignment = assignments[i]

                # pick random points in local mode if model is area and didn't receive any assignments
                # this case cannot happen in global mode (since it will always predict ALL samples)
                if len(assignment) == 0 and self._assign_random_points_if_model_outside: # only if requested by user
                    assignment = self.random_generator_.integers(0,
                                                            len(dsel_data),
                                                            max(2,
                                                                int(len(dsel_data) / len(models))))

            else: # global mode (
                assignment = assignments

            # Note that models eventually can die in local mode if we don't force them some point (which is optional)
            if len(assignment) > 0:
                # swap actual data to models data if applicaple
                if self.X_expert_dsel is not None:
                    data_to_evaluate = self.X_expert_dsel

                pred = model.predict(data_to_evaluate[assignment])
                correct_predict_idx = np.where(pred.astype(str) == dsel_labels[assignment].astype(str))[0]
                correct_predict_dsel = dsel_data[assignment][correct_predict_idx]

                if len(correct_predict_idx) > 0:
                    force = np.sum(correct_predict_dsel, axis=0) / len(correct_predict_idx)
                else:
                    force = np.zeros_like(pos)
                direction_to_force = force - pos

                direction_to_force *= step_size
                self.model_positions_[i] += direction_to_force

    def _calculate_performance(self, weight_val_data: float = 0.5) -> float:
        """
        Will calculate the performance in range 0..1
        using current configuration
        :param weight_val_data: How much the performance should be influenced by the validation data performance
        :return:
        """
        assert 0 <= weight_val_data <= 1, "The weight for validation data performance" \
                                          "has to be in range [0...1] (Code: 0378289348234)"
        if self.X_expert is None:
            res_train = self.predict(self.get_train_data(processed=False))
            res_val = None
            if self.val_data_ is not None:
                res_val = self.predict(self.get_val_data(processed=False))

        else:
            res_train = self.predict(X=self.get_train_data(processed=False), X_expert=self.X_expert)
            res_val = None
            if self.val_data_ is not None:
                res_val = self.predict(X=self.get_val_data(processed=False), X_expert=self.get_expert_data_val())

        # mix training accuracy and test accuracy
        acc_train = accuracy_score(y_true=self.get_train_targets(processed=False).astype(str),
                                   y_pred=res_train.astype(str))

        if res_val is not None:
            acc_val = accuracy_score(y_true=self.get_val_targets(processed=False).astype(str),
                                       y_pred=res_val.astype(str))
        else:
            acc_val = acc_train

        return (acc_val * weight_val_data) + ((1 - weight_val_data)  * acc_train)

    def calculate_expert_local_performance(self, expert_idx: int, assignment: Optional[np.ndarray]=None) -> float:
        """
        Calculates the performance of the given model over the assigned data

        :param expert_idx:
        :param assignment: If you already calculated assignments somewhere else you my optionally hand it over
        to save computational time
        :return:
        """
        assert 0 <= expert_idx < self.n_classifiers_, "Expert index must be between 0 and number of classifiers " \
                                                      "(Code: 3209480293)"

        if assignment is None:
            assignments = self.assign_data_points_to_model(X=self.get_DSEL(processed=True),
                                                           is_processed=True)
            model_assignment = assignments[expert_idx]
        else:
            model_assignment = assignment

        data_X = self.get_DSEL(processed=False)[model_assignment] if self.X_expert is None \
            else self.X_expert_dsel[model_assignment]

        preds = self.pool_classifiers_[expert_idx].predict(X=data_X)
        y_true = self.get_DSEL_target(processed=True)

        return accuracy_score(y_true=y_true[model_assignment].astype(str), y_pred=preds.astype(str))

    def calculate_expert_global_performance(self, expert_idx: int) -> float:
        """
        Calculates the performance of the given model over the whole data (including not assigned)

        :param expert_idx:
        :return:
        """
        assert 0 <= expert_idx < self.n_classifiers_, "Expert index must be between 0 and number of classifiers " \
                                                     "(Code: 2342342345)"

        data_X = self.get_DSEL(processed=False) if self.X_expert is None \
            else self.X_expert_dsel

        preds = self.pool_classifiers_[expert_idx].predict(X=data_X )
        y_true = self.get_DSEL_target(processed=True)

        return accuracy_score(y_true=y_true.astype(str),
                              y_pred=preds.astype(str))

    def predict(self, X: Union[List, np.ndarray], X_expert: Optional[Union[List, np.ndarray]]=None) -> np.ndarray:
        """
        Predicts using the current model

        :param X:
        :param prepare_data:
        :return:
        """
        if self.X_expert is not None and X_expert is None:
            raise Exception("The model was trained with separate data for experts. "
                            "You have to additionally provide data in the format"
                            " that the expert was trained on! (Code: 3980923423)")

        if X_expert is not None:
            if len(X) != len(X_expert):
                raise Exception("Expert data has to have the same amount of samples like X (Code: 8923478923)")

        assignments = self.assign_data_points_to_model(X=X)
        models = self.pool_classifiers_
        predictions = np.ndarray(shape=(len(X),),
                                 dtype=self.get_DSEL_target(processed=True).dtype)

        # predict each point by its closest model
        for model_idx, data_points in assignments.items():
            model = models[model_idx]

            if X_expert is None:
                X_p = self._preprocess_X(X=X)
                to_predict = X_p[data_points]
            else:
                to_predict = X_expert[data_points]

            if len(to_predict) > 0:
                predictions[data_points] = model.predict(X=to_predict)

        return self.enc_.inverse_transform(predictions)

    def get_current_classifiers(self):
        """
        Returns the current models inside
        :return:
        """
        return self.pool_classifiers_

    def get_important_feature_indexer(self) -> np.ndarray:
        """
        Gets feature indexer for indexing important features, i.e., attributes will be taken into
        account when calculating distances
        Initializes as a vector containing only `True` vals, if indexer was not set before
        :return:
        """

        if self.important_feature_indexer_ is None:
            # init feature selection
            self.important_feature_indexer_ = np.ndarray(shape=(self.get_data_dimensionality(),),
                                                         dtype=bool)
            self.important_feature_indexer_[:] = True

        return self.important_feature_indexer_

    def refit_nerds(self):
        """
        Will find closest models for each point and fits
        model on closest points, if you keep track of the models
        inside the pool, be aware that function might REPLACE them with new instances
        """
        self._fit_classifiers_to_closest_points()

    def _fit_classifiers_to_closest_points(self, only_expert_idx: Optional[int] = None):
        """
        Will find closest models for each point and fits
        model on closest points, if you keep track of the models
        inside the pool, be aware that function might REPLACE them with new instances

        """
        train_data_prepared = self.get_train_data(processed=True)
        train_targets_prepared = self.get_train_targets(processed=True)
        models = self.get_current_classifiers()
        model_to_points = self.assign_data_points_to_model(X=train_data_prepared,
                                                           is_processed=True)

        # if we have separate Expert data, we switch to training on those after we know which points belong to them
        if self.X_expert is not None:
            train_data_prepared = self.X_expert

        # collect assignments
        model_to_data_point_indices = {}
        for model_idx, assigned_data_indices in model_to_points.items():
            # if all data points are assigned somewhere else
            # we optionally pick some random points
            if len(assigned_data_indices) == 0 and self._assign_random_points_if_model_outside:
                assigned_data_indices = self.random_generator_.integers(0,
                                                                   len(train_data_prepared),
                                                                   max(2,
                                                                       int(len(train_data_prepared) / len(model_to_points))))

            model_to_data_point_indices[model_idx] = assigned_data_indices
        # actually fit the model
        if only_expert_idx is not None:  # if we only fit one we skip the parallel part
            self.pool_classifiers_[only_expert_idx].fit(
                train_data_prepared[model_to_data_point_indices[only_expert_idx]],
                train_targets_prepared[model_to_data_point_indices[only_expert_idx]])
        else:
            try:
                data = Parallel(n_jobs=min(model_idx + 1, self.get_max_jobs()))(delayed(_parallel_fit)(
                                                                                     model_cls= models[model_idx].__class__,
                                                                                     model_idx=model_idx,
                                                                                     model_params=models[model_idx].get_params(),
                                                                                     X=train_data_prepared[ass],
                                                                                     y=train_targets_prepared[ass])
                                                          for model_idx, ass in model_to_data_point_indices.items())
                # write models back
                for result_model, idx in data:
                    self.pool_classifiers_[idx] = result_model

            except Exception as e:
                self.log_warning(f"There was an error on parallel fit \"{e}\". "
                                 f"You should disable that feature. Will fall back"
                                 f" to iterative fit (Code: 90348290348)")

                # try sequential fit if parallel fails
                for model_idx, assigned_data_indices in model_to_points.items():
                    if not self.is_classifier_fixed(idx=model_idx):
                        try:

                            self.pool_classifiers_[model_idx].fit(train_data_prepared[model_to_data_point_indices[model_idx]],
                                                                 train_targets_prepared[model_to_data_point_indices[model_idx]])

                        except Exception as e:
                            # import traceback as tb
                            # print(tb.print_tb(e.__traceback__))
                            self.log_warning(f"Could not fit model with idx {model_idx} due to {e.__str__()}. "
                                             f"I Leave it alone atm (Code: 739792658)")


    def is_classifier_fixed(self, idx: int):
        """
        Returns True iff the classifier should not change in structure anymore (no refit!)
        :param idx:
        :return:
        """
        return self._fixed_classifiers[idx]

    def set_classifier_fixed(self, idx: int, fix: bool):
        """
        Fixes or unfixes the classifiers structure
        :param idx:
        :param fix:
        :return:
        """
        self._fixed_classifiers[idx] = fix


    def remove_model(self, model_idx: int, refit: bool=True) -> Tuple[BaseEstimator, np.ndarray]:
        """
        Will remove the model with the given index from the system and optionally refit
        the trees

        Returns the model and the position of the removed model

        :param model_idx:
        :param refit:
        :return:
        """
        assert 0 <= model_idx < len(self.pool_classifiers_), "The requested model index is not available " \
                                                             "(Code: 93480932)"

        model, pos = (self.pool_classifiers_[model_idx], self.model_positions_[model_idx])
        del self.pool_classifiers_[model_idx]
        del self.model_positions_[model_idx]
        del self._fixed_classifiers[model_idx]
        self.n_classifiers_ -= 1

        if refit is True:
            self.refit_nerds()

        return model, pos

    def calculate_model_distances_to_sample(self,
                                            X: Optional[np.ndarray] = None,
                                            is_processed: bool = False) -> np.ndarray:
        """
        Returns a sample-by-model index which contains the distances to each model (in order)
        to datapoints in X

        :param X:
        :param is_processed:
        :return:
        """
        # transform data to DSEL format
        if X is not None:
            X_dsel = self._preprocess_dsel(X=X) if not is_processed else X
        else:
            X_dsel = self.get_DSEL(processed=True)

        # read model points
        model_points = self.model_positions_

        # standard deviations (Determine the step size)
        stddev = self.dsel_std_

        # create a samples x models distance matrix
        distances = np.zeros(shape=(X_dsel.shape[0],
                                    model_points.shape[0]))

        # if we use feature importances, we use last calculate distance only over the important attributes
        if self._maximum_selected_features is not None:
            X_dsel = X_dsel[:, self.get_important_feature_indexer()]
            model_points = model_points[:, self.get_important_feature_indexer()]
            stddev = stddev[self.get_important_feature_indexer()]

        for model_idx in range(model_points.shape[0]):
            # p=1 minekowski distance normalized by stdev
            model_point = model_points[model_idx, :]
            distance = np.sum(np.abs((X_dsel - model_point)) / stddev,
                              axis=1)

            distances[:, model_idx] = distance

        return distances

    def assign_data_points_to_model(self, X: Optional[np.ndarray] = None,
                                    is_processed: bool = False,
                                    ) -> Dict[int, np.ndarray]:
        """
        Returns a dict that maps model indices their closest data points
        if X is `None` it will use DSEL data

        :return:
        """
        distances = self.calculate_model_distances_to_sample(X=X, is_processed=is_processed)
        # get row-wise  minimal distance
        closest_models = np.argmin(distances, axis=1)

        # construct distionary {model: [data_indices]
        d = {idx: np.argwhere(closest_models == idx)[:, 0] for idx in range(len(self.pool_classifiers_))}
        # d = {idx: (ass) if (ass := np.argwhere(closest_models == idx)[:, 0]) is not None else  [] for idx in range(len(model_points))}

        return d

    def get_DSEL(self, processed=True) -> np.ndarray:
        """
        Will return the DSEL data

        :param processed:
        :return:
        """
        assert self.DSEL_processed_ is not None, "DSEL data is not yet been set (propably " \
                                                 "the model is not fit) (Code: 328472389)"
        if processed:
            return self.DSEL_processed_

        return self.DSEL_data_

    def get_DSEL_target(self, processed=True) -> np.ndarray:
        """
        Will return the DSEL target

        :param processed:
        :return:
        """
        assert self.DSEL_processed_ is not None, "DSEL data is not yet been set (propably " \
                                                 "the model is not fit) (Code: 4564564564)"
        if processed:
            return self.DSEL_processed_target_

        return self.DSEL_target_

    def get_train_targets(self, processed=True) -> np.ndarray:
        """
          Will return the training targets / labels
          :return:
        """
        assert self.train_target_ is not None, "Training datas' labels is not yet been set (propably " \
                                               "the model is not fit) (Code: 4564564567)"

        if processed:
            return self.train_data_processed_target_

        return self.train_target_

    def get_val_targets(self, processed=True) -> np.ndarray:
        """
          Will return the training targets / labels
          :return:
        """
        assert self.val_target_ is not None, "Validation datas' labels is not yet been set (propably " \
                                             "the model is not fit) (Code: 3453434534694)"

        if processed:
            return self.val_data_processed_target_

        return self.val_target_

    def get_train_data(self, processed=True) -> np.ndarray:
        """
        Will return the training data (LoE version)
        :param processed:
        :return:
        """
        assert self.train_data_processed_ is not None, "Training data is not yet been set (propably " \
                                                       "the model is not fit) (Code: 4564564567)"
        if processed:
            return self.train_data_processed_
        else:
            return self.train_data_

    def get_val_data(self, processed=True) -> np.ndarray:
        """
        Will return the val data (LoE version)
        :param processed:
        :return:
        """
        assert self.val_data_processed_ is not None, "Validation data  is not yet been set (propably " \
                                                       "the model is not fit) (Code: 23847239847)"
        if processed:
            return self.val_data_processed_
        else:
            return self.val_data_

    def get_expert_data(self) -> np.ndarray:
            """
            Will return the training data (experts' version) if not set it will return the prepared train data

            :param processed:
            :return:
            """
            if self.X_expert is None:
                return self.get_train_data(processed=True)

            return self.X_expert

    def get_expert_data_val(self) -> np.ndarray:
        """
        Will return the validation data (experts' version)

        :param processed:
        :return:
        """
        assert self.X_expert_val is not None, "Expert data data is not yet been set (propably " \
                                              "the model is not fit) (Code: 4564564523)"
        return self.X_expert_val

    def get_data_dimensionality(self) -> int:
        """
        Gets the dimensionality of the data (DSEL data!)
        :return:
        """
        return self.get_DSEL().shape[1]

    def _init_model_positions(self) -> np.ndarray:
        """
        Will return initial points for the models in feature space
        :return:
        """
        pos = np.zeros(shape=(len(self.pool_classifiers_), self.get_data_dimensionality()), dtype=float)
        for i in range(len(self.get_current_classifiers())):
            pos[i, :] = self.generate_model_pos_in_feature_space()
        return pos

    def generate_model_pos_in_feature_space(self):
        """
        Will return a random position within feature spawn
        :return:
        """
        return [self.random_generator_.normal(loc=0,
                                              size=1,
                                              scale=self.dsel_std_[dim])[0] for dim in range(self.get_data_dimensionality())]

    def add_model(self, model: BaseEstimator):
        """
        Will add a new model within feature space and fits all models to (new) closest points

        :param model:
        :return:
        """
        pos = self.generate_model_pos_in_feature_space()
        pos = np.array(pos).reshape((1, self.get_data_dimensionality()))
        self.model_positions_ = np.append(self.model_positions_, pos, axis=0)
        self.n_classifiers_ += 1
        self.pool_classifiers_.append(model)
        self._fixed_classifiers.append(False)
        self._fit_classifiers_to_closest_points()

    def remove_model(self, model_idx: int):
        """
        Will remove a model from the pool
        and refit the remainder
        :param model_idx:
        :return:
        """
        assert model_idx >= 0 and model_idx < len(self.pool_classifiers_), \
            "Model index higher than models inside (Code: 23894723894)"

        assert self.n_classifiers_ > 1, "After removal there has to remain at least one model (Code: 2398742389)"

        del self.pool_classifiers_[model_idx]
        self.model_positions_ = np.delete(self.model_positions_, model_idx, axis=0)
        self.n_classifiers_ -= 1
        self._fit_classifiers_to_closest_points()

    def select(self, query: np.ndarray):
        super().select(query=query)
        assignments = self.assign_data_points_to_model(X=query, is_processed=False)
        selected_classifiers = np.ndarray(shape=(len(query),), dtype=np.object)
        for model_idx, assignment in enumerate(assignments):
            assignments[assignments] = self.get_current_classifiers()[model_idx]

        return selected_classifiers

    def pool_supports_feature_importances(self) -> bool:
        """
        Returns True if and only if all models support feature importances
        :return:
        """
        return all([hasattr(model, 'feature_importances_') for model in self.pool_classifiers_])

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Returns the feature gathered over whole ensemble
        Each models feature importance vector is summed up and weighted by the relative amount of samples
        assigned to the model

        this is mixed by the estimated importance of the features used to assign data to nerds (estimated
        by a rather big extra tree)
        :return:
        """
        assert self.pool_supports_feature_importances(), "Feature importances can only be calculated if all " \
                                                         "experts are able to return feature importances (Code: 435324)"

        # prepare array that holds importances for each classifier
        importances_nerds = np.ndarray(shape=(self.n_classifiers_,
                                        self.get_data_dimensionality()))

        n_samples = len(self.get_DSEL())
        assignments = self.assign_data_points_to_model(self.get_DSEL(processed=True), is_processed=True)
        # fill array row-wise with importances per model
        for i in range(0, self.n_classifiers_):

            # try to fetch the feature importances
            try:
                curr_imp = self.pool_classifiers_[i].feature_importances_
            except Exception as e:
                self.log_warning(f"Model at index {i} raised error \"{e}\" when trying to fetch feature importances."
                                 f"Will fall back to equal distribution for that model (Code: 347238974)")

                curr_imp = np.ndarray(shape=(self.get_data_dimensionality(), ), dtype=float)
                curr_imp[:] = 1. / self.get_data_dimensionality()

            if len(curr_imp) != self.get_data_dimensionality():
                raise Exception(f"Amount of feature importances returned by classifier {i} does not match "
                                f"data dimensionality. (Code: 348230948230)")

            # store RELATIVE importances (weighted by amount of samples) current row
            importances_nerds[i, :] = (len(assignments[i]) / n_samples) * self.pool_classifiers_[i].feature_importances_

        # sum up each features importance
        importances_nerds = np.sum(importances_nerds, axis=0)
        importances_nerds /= sum(importances_nerds)

        # calculate assignment importances
        labels = np.zeros(n_samples, dtype=int)
        for model_idx, assignment in assignments.items():
            labels[assignment] = model_idx

        # @TODO make these parameters or make the whole model an estimator
        et = ExtraTreesClassifier(n_estimators=1000,
                                  max_depth=6,                      # don't waste too much time
                                  random_state=self.random_state_)  # make this deterministic so that multiple calls
                                                                    # to feature importance doesn't change the output
        
        et.fit(self.get_DSEL(), labels)
        importances_et = et.feature_importances_

        # @TODO make the mixing a parameter
        return 0.5 * importances_nerds + 0.5 * importances_et

    @feature_importances_.setter
    def feature_importances_(self, value):
        """
        Feature importances are not meant to be set, since theyre defined by the
        models inside. This is a noop
        :param value:
        :return:
        """
        self.log_debug("You are trying to set feature importances. This is a calculated property defined by"
                       "the experts. Hence, request results in a noop (Code: 32423094238)")

    def set_maximum_selected_features(self, n_features: Optional[int]=None):
        """
        Sets the maximum amount of features that are taken into account while moving the model
        if None, we use all features
        :param n_features:
        :return:
        """
        if n_features is None:
            self.important_feature_indexer_ = None
            self.current_feature_importances_ = None

        self._maximum_selected_features = n_features

    def auto_determine_maximum_selected_features(self, include_features_until_information: float=0.6):
        """
        Will set the @see self._maximum_selected_features to an amount that satisfies include_features_until_information
        units of information
        :return:
        """

        assert 0 < include_features_until_information <= 1, "included information must be in ]0, 1] (Code: 349823904)"

        # just collect features until we explain more than 60% of the importance
        tree = ExtraTreesClassifier(n_estimators=1000, max_depth=3,
                                    random_state=self.random_state_,
                                    bootstrap=True,
                                    )
        tree.fit(self.get_train_data(processed=False), self.get_train_targets(processed=False))

        importances = sorted(tree.feature_importances_, reverse=True)
        score: float = 0
        for i, imp in enumerate(importances):
            score += imp
            if score >= include_features_until_information:
                break

        self._maximum_selected_features = i + 1

    def get_maximum_selected_features(self) -> Optional[Union[int, str]]:
        """
        Will return the maximum selected features for Nerd movement. This could be None if we donÄt restrict
        or auto if the value is to be determined before the first fit.
        This value should not be auto after training

        :return:
        """
        return self._maximum_selected_features
