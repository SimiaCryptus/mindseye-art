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

public @com.simiacryptus.ref.lang.RefAware
class TruncatedAffinity extends AffinityWrapper {
  private double min = 1e-2;
  private double power = 0.5;

  public TruncatedAffinity(RasterAffinity inner) {
    super(inner);
  }

  public double getMin() {
    return min;
  }

  public TruncatedAffinity setMin(double min) {
    this.min = min;
    return this;
  }

  public double getPower() {
    return power;
  }

  public TruncatedAffinity setPower(double power) {
    this.power = power;
    return this;
  }

  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> affinityList(
      com.simiacryptus.ref.wrappers.RefList<int[]> graphEdges) {
    final com.simiacryptus.ref.wrappers.RefList<double[]> innerResult = inner.affinityList(graphEdges);
    final double[] degree = RasterAffinity.degree(innerResult);
    final com.simiacryptus.ref.wrappers.RefList<double[]> doubles = RasterAffinity.adjust(graphEdges, innerResult,
        degree, getPower());
    final com.simiacryptus.ref.wrappers.RefList<double[]> truncated = doubles.stream()
        .map(list -> com.simiacryptus.ref.wrappers.RefArrays.stream(list).map(x -> x >= getMin() ? x : 0.0).toArray())
        .collect(com.simiacryptus.ref.wrappers.RefCollectors.toList());
    return RasterAffinity.adjust(graphEdges, truncated, degree, -getPower());
  }
}
