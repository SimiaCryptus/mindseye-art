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

import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;

public class TruncatedAffinity extends AffinityWrapper {
  private double min = 1e-2;
  private double power = 0.5;

  public TruncatedAffinity(RasterAffinity inner) {
    super(inner);
  }

  public double getMin() {
    return min;
  }

  @Nonnull
  public TruncatedAffinity setMin(double min) {
    this.min = min;
    return this;
  }

  public double getPower() {
    return power;
  }

  @Nonnull
  public TruncatedAffinity setPower(double power) {
    this.power = power;
    return this;
  }

  @Override
  public RefList<double[]> affinityList(@Nonnull RefList<int[]> graphEdges) {
    final RefList<double[]> innerResult = inner.affinityList(graphEdges);
    final double[] degree = RasterAffinity.degree(innerResult);
    final RefList<double[]> doubles = RasterAffinity.adjust(graphEdges, innerResult, degree, getPower());
    final RefList<double[]> truncated = doubles.stream()
        .map(list -> RefArrays.stream(list).map(x -> x >= getMin() ? x : 0.0).toArray())
        .collect(RefCollectors.toList());
    return RasterAffinity.adjust(graphEdges, truncated, degree, -getPower());
  }
}
