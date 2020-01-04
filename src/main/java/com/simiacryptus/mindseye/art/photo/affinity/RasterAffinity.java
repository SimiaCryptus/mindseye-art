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

import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

public @com.simiacryptus.ref.lang.RefAware
interface RasterAffinity {
  static com.simiacryptus.ref.wrappers.RefList<double[]> normalize(
      com.simiacryptus.ref.wrappers.RefList<int[]> graphEdges,
      com.simiacryptus.ref.wrappers.RefList<double[]> affinityList) {
    return adjust(graphEdges, affinityList, degree(affinityList), 0.5);
  }

  static com.simiacryptus.ref.wrappers.RefList<double[]> adjust(List<int[]> graphEdges,
                                                                List<double[]> affinityList, double[] degree, double power) {
    return com.simiacryptus.ref.wrappers.RefIntStream.range(0, graphEdges.size()).mapToObj(i2 -> {
      final double deg_i = degree[i2];
      final int[] edges = graphEdges.get(i2);
      final double[] affinities = affinityList.get(i2);
      if (deg_i == 0)
        return new double[edges.length];
      return com.simiacryptus.ref.wrappers.RefIntStream.range(0, edges.length).mapToDouble(j -> {
        final double deg_j = degree[edges[j]];
        if (deg_j == 0)
          return 0;
        return affinities[j] / Math.pow(deg_j * deg_i, power);
      }).toArray();
    }).collect(com.simiacryptus.ref.wrappers.RefCollectors.toList());
  }

  static double[] degree(List<double[]> affinityList) {
    return affinityList.stream().parallel().mapToDouble(x -> com.simiacryptus.ref.wrappers.RefArrays.stream(x).sum())
        .toArray();
  }

  default AffinityWrapper wrap(
      BiFunction<com.simiacryptus.ref.wrappers.RefList<int[]>, com.simiacryptus.ref.wrappers.RefList<double[]>, com.simiacryptus.ref.wrappers.RefList<double[]>> fn) {
    return new AffinityWrapper(this) {
      @Override
      public com.simiacryptus.ref.wrappers.RefList<double[]> affinityList(
          com.simiacryptus.ref.wrappers.RefList<int[]> graphEdges) {
        return fn.apply(graphEdges, inner.affinityList(graphEdges));
      }
    };
  }

  default AffinityWrapper wrap(
      Function<com.simiacryptus.ref.wrappers.RefList<int[]>, com.simiacryptus.ref.wrappers.RefList<double[]>> fn) {
    return new AffinityWrapper(this) {
      @Override
      public com.simiacryptus.ref.wrappers.RefList<double[]> affinityList(
          com.simiacryptus.ref.wrappers.RefList<int[]> graphEdges) {
        return fn.apply(graphEdges);
      }
    };
  }

  com.simiacryptus.ref.wrappers.RefList<double[]> affinityList(com.simiacryptus.ref.wrappers.RefList<int[]> graphEdges);

}
