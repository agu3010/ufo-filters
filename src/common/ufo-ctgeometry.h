/*
 * Copyright (C) 2017 Karlsruhe Institute of Technology
 *
 * This file is part of Ufo.
 *
 * This library is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef UFO_CTGEOMETRY_H
#define UFO_CTGEOMETRY_H

#include <ufo/ufo.h>
#include "ufo-scarray.h"

typedef struct {
    UfoScarray *x;
    UfoScarray *y;
    UfoScarray *z;
} UfoPoint;

typedef struct {
    UfoPoint *position;
    UfoPoint *angle;
} UfoVector;

typedef struct {
    UfoPoint  *source_position;
    UfoPoint  *volume_angle;
    UfoVector *axis;
    UfoVector *detector;
} UfoCTGeometry;

UfoPoint        *ufo_point_new             (UfoScarray *x,
                                            UfoScarray *y,
                                            UfoScarray *z);
UfoPoint        *ufo_point_copy            (const UfoPoint *point);
void             ufo_point_free            (UfoPoint *point);
gboolean         ufo_point_are_almost_zero (UfoPoint *point);
UfoVector       *ufo_vector_new            (UfoPoint *position,
                                            UfoPoint *angle);
void             ufo_vector_free           (UfoVector *vector);
UfoCTGeometry   *ufo_ctgeometry_new        (void);
void             ufo_ctgeometry_free       (UfoCTGeometry *geometry);

#endif
