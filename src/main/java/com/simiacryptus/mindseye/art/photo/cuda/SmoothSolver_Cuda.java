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

package com.simiacryptus.mindseye.art.photo.cuda;

import com.simiacryptus.mindseye.art.photo.SmoothSolver;
import com.simiacryptus.mindseye.art.photo.affinity.RasterAffinity;
import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.lang.Tensor;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class SmoothSolver_Cuda implements SmoothSolver {

  public static @NotNull CudaSparseMatrix laplacian(RasterAffinity affinity, RasterTopology topology) {
    return laplacian(
        topology.connectivity(),
        affinity.affinityList(topology.connectivity()));
  }

  public static @NotNull CudaSparseMatrix laplacian(List<int[]> graphEdges, List<double[]> affinityList) {
    final int pixels = graphEdges.size();
    final double[] doubles = RasterAffinity.normalize(graphEdges, affinityList).stream()
        .flatMapToDouble(x -> Arrays.stream(x)).toArray();
    return new CudaSparseMatrix(
        new SparseMatrixFloat(
            IntStream.range(0, pixels).flatMap(i1 -> Arrays.stream(graphEdges.get(i1))).toArray(),
            IntStream.range(0, pixels).flatMap(i -> Arrays.stream(graphEdges.get(i)).map(j -> i)).toArray(),
            SparseMatrixFloat.toFloat(doubles),
            pixels,
            pixels
        ).sortAndPrune().assertSymmetric());
  }

  @Override
  public RefOperator<Tensor> solve(RasterTopology topology, RasterAffinity affinity, double lambda) {
    double alpha = 1.0 / (1.0 + lambda);
    final CudaSparseMatrix laplacian = laplacian(affinity, topology);
    final SparseMatrixFloat forwardMatrix = forwardMatrix(laplacian, alpha);
    laplacian.freeRef();
    return new TensorOperator(new SingleChannelWrapper(new CudaMatrixSolver(forwardMatrix, 1 - alpha)), topology.getDimensions(), topology);
  }

  public SparseMatrixFloat forwardMatrix(@NotNull CudaSparseMatrix laplacian, double alpha) {
    return SparseMatrixFloat.identity(laplacian.matrix.rows).minus(laplacian.matrix.scale(alpha));
  }

}
