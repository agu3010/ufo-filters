/*
 * Copyright (C) 2015-2017 Karlsruhe Institute of Technology
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

#ifndef UFO_CONEBEAM_H
#define UFO_CONEBEAM_H

#define EXTRACT_INT(region, index) g_value_get_int (g_value_array_get_nth ((region), (index)))
#define EXTRACT_FLOAT(region, index) g_value_get_float (g_value_array_get_nth ((region), (index)))
#define EXTRACT_DOUBLE(region, index) g_value_get_double (g_value_array_get_nth ((region), (index)))
#define REGION_SIZE(region) ((EXTRACT_INT ((region), 2)) == 0) ? 0 : \
                            ((EXTRACT_INT ((region), 1) - EXTRACT_INT ((region), 0) - 1) /\
                            EXTRACT_INT ((region), 2) + 1)
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define IS_BEAM_PARALLEL(source_position) (isinf (EXTRACT_FLOAT ((source_position), 1)))
#define NEXT_DIVISOR(dividend, divisor) ((dividend) + (divisor) - (dividend) % (divisor))

#include <glib.h>
#include <glib-object.h>

typedef struct {
    gchar *name;
    guint burst;
    guint max_regcount;
} UniRecoNodeProps;

gfloat get_float_from_array_or_scalar (GValueArray *array,
                                       guint index);
gdouble get_double_from_array_or_scalar (GValueArray *array,
                                         guint index);
GHashTable *get_node_props_table (void);

#endif
