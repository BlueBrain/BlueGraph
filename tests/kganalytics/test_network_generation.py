#
# Blue Brain Graph is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Blue Brain Graph is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Blue Brain Graph. If not, see <https://choosealicense.com/licenses/lgpl-3.0/>.

import pytest

from kganalytics.network_generation import mutual_information

def test_mutual_information(occurrence_data):

    with pytest.raises(ValueError):
        mutual_information(occurrence_data[0], "paper", 100,
                           "virus", "a34r", mitype="non_mitype")