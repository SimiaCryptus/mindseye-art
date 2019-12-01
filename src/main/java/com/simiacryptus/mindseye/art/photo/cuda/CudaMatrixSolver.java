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

import com.simiacryptus.ref.lang.ReferenceCountingBase;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcusolver.JCusolver;
import jcuda.jcusolver.cusolverSpHandle;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.runtime.JCuda;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.stream.IntStream;

import static jcuda.jcusolver.JCusolverSp.cusolverSpScsrlsvchol;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class CudaMatrixSolver extends ReferenceCountingBase implements RefOperator<double[][]> {

  public static final double TOLERANCE = 1e-8;
  private static final int REORDER = 0;

  final double scaleOutput;
  final CudaSparseMatrix forwardMatrix;
  final int pixels;
  @NotNull cusparseHandle spHandle;
  private @NotNull
  cusolverSpHandle solverHandle;

  public CudaMatrixSolver(SparseMatrixFloat forwardMatrix) {
    this(forwardMatrix, 1.0);
  }

  public CudaMatrixSolver(SparseMatrixFloat forwardMatrix, double scaleOutput) {
    JCuda.setExceptionsEnabled(true);
    JCusparse.setExceptionsEnabled(true);
    JCusolver.setExceptionsEnabled(true);
    this.scaleOutput = scaleOutput;
    this.forwardMatrix = new CudaSparseMatrix(forwardMatrix);
    solverHandle = CudaSparseMatrix.newSolverHandle();
    spHandle = CudaSparseMatrix.newSparseHandle();
    pixels = forwardMatrix.cols;
  }

  @Override
  protected void _free() {
    forwardMatrix.freeRef();
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
    final CudaSparseMatrix.GpuCopy gpuCopy = forwardMatrix.get();
    cusolverSpScsrlsvchol(
        solverHandle,
        pixels,
        forwardMatrix.matrix.getNumNonZeros(),
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
        IntStream.range(0, pixels).mapToDouble(i -> scaleOutput * floats[c * pixels + i]).toArray()
    ).toArray(i -> new double[i][]);
  }
}
