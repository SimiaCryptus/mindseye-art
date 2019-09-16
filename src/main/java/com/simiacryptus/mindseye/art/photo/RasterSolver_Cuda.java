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

import com.simiacryptus.lang.ref.ReferenceCountingBase;
import com.simiacryptus.mindseye.lang.Tensor;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcusolver.JCusolver;
import jcuda.jcusolver.cusolverSpHandle;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.runtime.JCuda;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import static jcuda.jcusolver.JCusolverSp.cusolverSpScsrlsvchol;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class RasterSolver_Cuda implements RasterSolver {

  private static final int REORDER = 0;
  private static final double TOLERANCE = 1e-8;

  @Override
  @NotNull
  public RefOperator<Tensor> smoothingTransform(double lambda, RasterAffinity affinity) {
    RasterTopology topology = affinity.getTopology();
    double alpha = 1.0 / (1.0 + lambda);
    return new TensorOperator(new SingleChannelWrapper(new RawSolution(forwardMatrix(laplacian(affinity), alpha), alpha)), topology.getDimensions(), topology);
  }

  public SparseMatrix forwardMatrix(@NotNull CudaSparseMatrix laplacian, double alpha) {
    final SparseMatrix identity = SparseMatrix.identity(laplacian.matrix.rows);
    final SparseMatrix scale = laplacian.matrix.scale(alpha);
    final SparseMatrix matrix = identity.minus(scale);
    return matrix;
  }

  public @NotNull CudaSparseMatrix laplacian(RasterAffinity affinity) {
    return laplacian(
        affinity.getTopology().connectivity(),
        affinity.affinityList(affinity.getTopology().connectivity()));
  }

  public @NotNull CudaSparseMatrix laplacian(List<int[]> graphEdges, List<double[]> affinityList) {
    final int pixels = graphEdges.size();
    return new CudaSparseMatrix(
        new SparseMatrix(IntStream.range(0, pixels).flatMap(i1 -> Arrays.stream(graphEdges.get(i1))).toArray(), IntStream.range(0, pixels).flatMap(i -> Arrays.stream(graphEdges.get(i)).map(j -> i)).toArray(), SparseMatrix.toFloat(RasterAffinity.normalize(graphEdges, affinityList).stream().flatMapToDouble(x -> Arrays.stream(x)).toArray()), pixels, pixels));
  }

  private static class SingleChannelWrapper extends ReferenceCountingBase implements RefOperator<double[][]> {
    private final RefOperator<double[][]> unaryOperator;

    public SingleChannelWrapper(RefOperator<double[][]> unaryOperator) {
      this.unaryOperator = unaryOperator;
    }

    @Override
    public double[][] apply(double[][] img) {
      return Arrays.stream(img)
          .map(x -> unaryOperator.apply(new double[][]{x})[0])
          .toArray(i -> new double[i][]);
    }

    @Override
    protected void _free() {
      this.unaryOperator.freeRef();
      super._free();
    }
  }

  private static class TensorOperator extends ReferenceCountingBase implements RefOperator<Tensor> {
    private final RefOperator<double[][]> inner;
    private final int[] dimensions;
    private final RasterTopology topology;

    public TensorOperator(RefOperator<double[][]> inner, int[] dimensions, RasterTopology topology) {
      this.inner = inner;
      this.dimensions = dimensions;
      this.topology = topology;
    }

    @Override
    public Tensor apply(Tensor tensor) {
      if (!Arrays.equals(dimensions, tensor.getDimensions()))
        throw new IllegalArgumentException(Arrays.toString(dimensions) + " != " + Arrays.toString(tensor.getDimensions()));
      final double[][] imageMatrix = IntStream.range(0, dimensions[2]).mapToObj(c -> {
        final double[] doubles = new double[dimensions[0] * dimensions[1]];
        for (int y = 0; y < dimensions[1]; y++) {
          for (int x = 0; x < dimensions[0]; x++) {
            doubles[topology.getIndexFromCoords(x, y)] = tensor.get(x, y, c);
          }
        }
        return doubles;
      }).toArray(i -> new double[i][]);
      double[][] smoothed = inner.apply(imageMatrix);
      return tensor.mapCoords(coordinate -> {
        final int[] c = coordinate.getCoords();
        return Math.min(Math.max(smoothed[c[2]][topology.getIndexFromCoords(c[0], c[1])], 0), 255);
      });
    }

    @Override
    protected void _free() {
      inner.freeRef();
      super._free();
    }

  }

  private class RawSolution extends ReferenceCountingBase implements RefOperator<double[][]> {
    final double alpha;
    final CudaSparseMatrix laplacian;
    final int pixels;
    @NotNull cusparseHandle spHandle;
    private @NotNull
    cusolverSpHandle solverHandle;

    public RawSolution(SparseMatrix laplacian, double alpha) {
      JCuda.setExceptionsEnabled(true);
      JCusparse.setExceptionsEnabled(true);
      JCusolver.setExceptionsEnabled(true);
      this.alpha = alpha;
      this.laplacian = new CudaSparseMatrix(laplacian);
      solverHandle = CudaSparseMatrix.newSolverHandle();
      spHandle = CudaSparseMatrix.newSparseHandle();
      pixels = laplacian.cols;
    }

    @Override
    protected void _free() {
      laplacian.freeRef();
      super._free();
    }

    @Override
    public double[][] apply(double[][] img) {
      final int channels = img.length;
      final float[] flattened = new float[Arrays.stream(img).mapToInt(x -> x.length).sum()];
      for (int i = 0; i < flattened.length; i++) {
        flattened[i] = (float) img[i / pixels][i % pixels];
      }
      int singularityRowArray[] = {-1};
      Pointer gpuResult = new Pointer();
      cudaMalloc(gpuResult, Sizeof.FLOAT * flattened.length);
      final Pointer input = CudaSparseMatrix.toDevice(flattened);
      final CudaSparseMatrix.GpuCopy gpuCopy = laplacian.get();
      cusolverSpScsrlsvchol(
          solverHandle,
          pixels,
          laplacian.matrix.getNumNonZeros(),
          CudaSparseMatrix.descriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO),
          gpuCopy.values,
          gpuCopy.csrRows(spHandle),
          gpuCopy.columnIndices,
          input,
          (float) TOLERANCE,
          REORDER,
          gpuResult,
          singularityRowArray);
      gpuCopy.freeRef();
      if (singularityRowArray[0] != -1) System.err.println("Singular pixel: " + singularityRowArray[0]);
      cudaFree(input);

      final float[] floats = new float[channels * pixels];
      cudaMemcpy(Pointer.to(floats), gpuResult, (long) channels * Sizeof.FLOAT * pixels, cudaMemcpyDeviceToHost);
      cudaFree(gpuResult);

      return IntStream.range(0, channels).mapToObj(c ->
          IntStream.range(0, pixels).mapToDouble(i -> (1 - alpha) * floats[c * pixels + i]).toArray()
      ).toArray(i -> new double[i][]);
    }
  }
}
