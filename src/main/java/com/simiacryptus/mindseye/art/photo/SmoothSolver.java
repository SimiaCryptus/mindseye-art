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

package com.simiacryptus.mindseye.art.photo;

import com.simiacryptus.mindseye.art.photo.affinity.RasterAffinity;
import com.simiacryptus.mindseye.art.photo.cuda.RefUnaryOperator;
import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.RefAware;

import javax.annotation.Nonnull;

/**
 * The interface Smooth solver.
 */
public interface SmoothSolver {
  /**
   * Solve ref unary operator.
   *
   * @param topology the topology
   * @param affinity the affinity
   * @param lambda   the lambda
   * @return the ref unary operator
   */
  @Nonnull
  RefUnaryOperator<Tensor> solve(RasterTopology topology, @RefAware RasterAffinity affinity, double lambda);
}
