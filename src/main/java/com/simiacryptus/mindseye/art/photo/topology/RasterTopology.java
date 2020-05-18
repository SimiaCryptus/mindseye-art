/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.art.photo.topology;

import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.ReferenceCounting;

import javax.annotation.Nonnull;
import java.util.List;

/**
 * The interface Raster topology.
 */
public interface RasterTopology extends ReferenceCounting {

  /**
   * Get dimensions int [ ].
   *
   * @return the int [ ]
   */
  int[] getDimensions();

  /**
   * Cached raster topology.
   *
   * @return the raster topology
   */
  @Nonnull
  @RefAware
  default RasterTopology cached() {
    return new RasterTopologyWrapper.CachedRasterTopology(this);
  }

  /**
   * Connectivity list.
   *
   * @return the list
   */
  List<int[]> connectivity();

  /**
   * Gets index from coords.
   *
   * @param x the x
   * @param y the y
   * @return the index from coords
   */
  int getIndexFromCoords(int x, int y);

  /**
   * Get coords from index int [ ].
   *
   * @param i the
   * @return the int [ ]
   */
  int[] getCoordsFromIndex(int i);

  @Override
  RasterTopology addRef();
}