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

import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

public interface RasterAffinity {
  static RefList<double[]> normalize(RefList<int[]> graphEdges, RefList<double[]> affinityList) {
    return adjust(graphEdges, affinityList, degree(affinityList), 0.5);
  }

  static RefList<double[]> adjust(List<int[]> graphEdges, List<double[]> affinityList, double[] degree, double power) {
    return RefIntStream.range(0, graphEdges.size()).mapToObj(i2 -> {
      final double deg_i = degree[i2];
      final int[] edges = graphEdges.get(i2);
      final double[] affinities = affinityList.get(i2);
      if (deg_i == 0)
        return new double[edges.length];
      return RefIntStream.range(0, edges.length).mapToDouble(j -> {
        final double deg_j = degree[edges[j]];
        if (deg_j == 0)
          return 0;
        return affinities[j] / Math.pow(deg_j * deg_i, power);
      }).toArray();
    }).collect(RefCollectors.toList());
  }

  static double[] degree(List<double[]> affinityList) {
    return affinityList.stream().parallel().mapToDouble(x -> RefArrays.stream(x).sum()).toArray();
  }

  default AffinityWrapper wrap(BiFunction<RefList<int[]>, RefList<double[]>, RefList<double[]>> fn) {
    return new AffinityWrapper(this) {
      @Override
      public RefList<double[]> affinityList(RefList<int[]> graphEdges) {
        return fn.apply(graphEdges, inner.affinityList(graphEdges));
      }
    };
  }

  default AffinityWrapper wrap(Function<RefList<int[]>, RefList<double[]>> fn) {
    return new AffinityWrapper(this) {
      @Override
      public RefList<double[]> affinityList(RefList<int[]> graphEdges) {
        return fn.apply(graphEdges);
      }
    };
  }

  RefList<double[]> affinityList(RefList<int[]> graphEdges);

}
