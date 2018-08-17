// Copyright(c) 2018 The University of Edinburgh
//
// This file is part of tlm_adjoint.
// 
// tlm_adjoint is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, version 3 of the License.
// 
// tlm_adjoint is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

Point(1) = {0, 0, 0, 0.02};
Point(2) = {2, 0, 0, 0.02};
Point(3) = {2, 1, 0, 0.02};
Point(4) = {0, 1, 0, 0.02};
Line(1) = {1, 2};
Line(2) = {3, 2};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(6) = {3, 4, 1, -2};
Plane Surface(7) = {6};
Physical Surface(8) = {7};
