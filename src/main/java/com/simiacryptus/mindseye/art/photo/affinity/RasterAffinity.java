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

package com.simiacryptus.mindseye.art.photo.affinity;

import com.simiacryptus.mindseye.lang.CoreSettings;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The interface Raster affinity.
 */
public interface RasterAffinity {
  /**
   * Normalize list.
   *
   * @param graphEdges   the graph edges
   * @param affinityList the affinity list
   * @return the list
   */
  static List<double[]> normalize(@Nonnull List<int[]> graphEdges, @Nonnull List<double[]> affinityList) {
    return adjust(graphEdges, affinityList, degree(affinityList), 0.5);
  }

  /**
   * Adjust list.
   *
   * @param graphEdges   the graph edges
   * @param affinityList the affinity list
   * @param degree       the degree
   * @param power        the power
   * @return the list
   */
  static List<double[]> adjust(@Nonnull List<int[]> graphEdges, @Nonnull List<double[]> affinityList, double[] degree, double power) {
    return IntStream.range(0, graphEdges.size()).mapToObj(i2 -> {
      final double deg_i = degree[i2];
      final int[] edges = graphEdges.get(i2);
      final double[] affinities = affinityList.get(i2);
      if (deg_i == 0)
        return new double[edges.length];
      return IntStream.range(0, edges.length).mapToDouble(j -> {
        final double deg_j = degree[edges[j]];
        if (deg_j == 0)
          return 0;
        return affinities[j] / Math.pow(deg_j * deg_i, power);
      }).toArray();
    }).collect(Collectors.toList());
  }

  /**
   * Degree double [ ].
   *
   * @param affinityList the affinity list
   * @return the double [ ]
   */
  static double[] degree(@Nonnull List<double[]> affinityList) {
    Stream<double[]> stream = affinityList.stream();
    if (!CoreSettings.INSTANCE().singleThreaded) stream = stream.parallel();
    return stream.mapToDouble(x -> Arrays.stream(x).sum()).toArray();
  }

  /**
   * Wrap affinity wrapper.
   *
   * @param fn the fn
   * @return the affinity wrapper
   */
  @Nonnull
  default AffinityWrapper wrap(@Nonnull BiFunction<List<int[]>, List<double[]>, List<double[]>> fn) {
    return new AffinityWrapper(this) {
      @Override
      public List<double[]> affinityList(List<int[]> graphEdges) {
        return fn.apply(graphEdges, inner.affinityList(graphEdges));
      }
    };
  }

  /**
   * Wrap affinity wrapper.
   *
   * @param fn the fn
   * @return the affinity wrapper
   */
  @Nonnull
  default AffinityWrapper wrap(@Nonnull Function<List<int[]>, List<double[]>> fn) {
    return new AffinityWrapper(this) {
      @Override
      public List<double[]> affinityList(List<int[]> graphEdges) {
        return fn.apply(graphEdges);
      }
    };
  }

  /**
   * Affinity list list.
   *
   * @param graphEdges the graph edges
   * @return the list
   */
  List<double[]> affinityList(List<int[]> graphEdges);

}
