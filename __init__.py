"""
# Copyright 2020 Professorship Media Informatics, University of Applied Sciences Mittweida
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
# @created: 02.04.2020
"""

from loe.classes.loe import LoE, CustomDCSBase, BaseEstimator
from loe.utils.base_gon_step_callback import BaseLoeStepCallBack, ScatterVideoCreatorLoE
import warnings


class DeprecationBase:
    def __new__(cls, *args, **kwargs):
        warnings.warn(cls.__name__ + " is deprecated. Do not use it anymore", DeprecationWarning)
        return super().__new__(cls)

class GonDeprecated(LoE, DeprecationBase):
    pass


class ScatterVideoCreatorGONDeprecated(ScatterVideoCreatorLoE, DeprecationBase):
    pass

class BaseGonStepCallbackDeprecated(BaseLoeStepCallBack, DeprecationBase):
    pass

# only for backwards compatibility. You should not use those in new projects
GON = GonDeprecated
ScatterVideoCreatorGON = ScatterVideoCreatorGONDeprecated
BaseGonStepCallBack = BaseGonStepCallbackDeprecated

__all__ = ['GON',
           'LoE',
           'CustomDCSBase',
           'BaseEstimator',
           'BaseLoeStepCallBack',
           'BaseGonStepCallBack',
           'ScatterVideoCreatorGON',
           'ScatterVideoCreatorLoE']