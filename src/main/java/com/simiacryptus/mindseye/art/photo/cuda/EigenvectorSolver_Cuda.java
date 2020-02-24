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

import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.FastRandom;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcusolver.JCusolver;
import jcuda.jcusolver.cusolverSpHandle;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.runtime.JCuda;

import javax.annotation.Nonnull;
import java.util.concurrent.atomic.AtomicReference;

import static jcuda.jcusolver.JCusolverSp.cusolverSpScsreigvsi;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class EigenvectorSolver_Cuda extends ReferenceCountingBase implements RefOperator<double[][]> {
  @Nonnull
  final CudaSparseMatrix laplacian;
  final int pixels;
  @Nonnull
  cusparseHandle spHandle;
  private @Nonnull
  cusolverSpHandle solverHandle;
  private float mu0;

  public EigenvectorSolver_Cuda(@Nonnull CudaSparseMatrix laplacian, float mu0) {
    JCuda.setExceptionsEnabled(true);
    JCusparse.setExceptionsEnabled(true);
    JCusolver.setExceptionsEnabled(true);
    this.laplacian = laplacian;
    solverHandle = CudaSparseMatrix.newSolverHandle();
    spHandle = CudaSparseMatrix.newSparseHandle();
    pixels = this.laplacian.matrix.cols;
    this.mu0 = mu0;
  }

  @Nonnull
  public static int[] get2D(@Nonnull RasterTopology topology) {
    return get2D(topology.getDimensions());
  }

  @Nonnull
  public static int[] get2D(int[] dimensions) {
    return new int[]{dimensions[0], dimensions[1]};
  }

  @Nonnull
  public static Tensor remaining(@Nonnull RefCollection<Tensor> eigenVectors, int... dimensions) {
    Tensor tensor = new Tensor(dimensions);
    tensor.set(() -> FastRandom.INSTANCE.random());
    AtomicReference<Tensor> seed = new AtomicReference<>(tensor.unit());
    tensor.freeRef();
    eigenVectors.forEach(eigenVector -> {
      Tensor prev = seed.get();
      final double dot = eigenVector.dot(prev.addRef());
      if (Double.isFinite(dot)) {
        seed.set(prev.minus(eigenVector.scale(dot)));
      }
      eigenVector.freeRef();
      prev.freeRef();
    });
    eigenVectors.freeRef();
    Tensor tensor1 = seed.get();
    Tensor unit = tensor1.unit();
    tensor1.freeRef();
    return unit;
  }

  @Nonnull
  public RefList<Tensor> eigenVectors(@Nonnull RasterTopology topology, int n) {
    final RefList<Tensor> eigenVectors = new RefArrayList<>();
    final int[] dimensions = topology.getDimensions();
    final TensorOperator eigenSolver = eigenRefiner(topology);
    for (int i = 0; i < n; i++) {
      final Tensor remaining = remaining(eigenVectors.addRef(), dimensions[0], dimensions[1]);
      eigenVectors.add(eigenSolver.apply(remaining));
    }
    eigenSolver.freeRef();
    //eigenVectors.add(remaining(eigenVectors, dimensions[0], dimensions[1]));
    return eigenVectors;
  }

  @Nonnull
  public TensorOperator eigenRefiner(@Nonnull RasterTopology topology) {
    return new TensorOperator(this, get2D(topology), topology);
  }

  public void _free() {
    laplacian.freeRef();
    super._free();
  }

  @Nonnull
  public double[][] apply(@Nonnull double[][] img) {
    final int channels = img.length;
    final float[] flattened = new float[RefArrays.stream(img).mapToInt(x -> x.length).sum()];
    for (int i = 0; i < flattened.length; i++) {
      flattened[i] = (float) img[i / pixels][i % pixels];
    }
    Pointer gpuResult = new Pointer();
    cudaMalloc(gpuResult, Sizeof.FLOAT * flattened.length);
    final Pointer input = CudaSparseMatrix.toDevice(flattened);
    final CudaSparseMatrix.GpuCopy laplacian_gpu = laplacian.get();
    final float[] mu = {mu0};
    final Pointer mu_out = CudaSparseMatrix.toDevice(mu);
    assert laplacian_gpu != null;
    cusolverSpScsreigvsi(solverHandle, pixels, laplacian.matrix.getNumNonZeros(),
        CudaSparseMatrix.descriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO), laplacian_gpu.values,
        laplacian_gpu.csrRows(spHandle), laplacian_gpu.columnIndices, mu[0], input, 100,
        (float) CudaMatrixSolver.TOLERANCE, mu_out, gpuResult);
    laplacian_gpu.freeRef();
    cudaFree(input);
    cudaMemcpy(Pointer.to(mu), mu_out, Sizeof.FLOAT, cudaMemcpyDeviceToHost);

    final float[] floats = new float[channels * pixels];
    cudaMemcpy(Pointer.to(floats), gpuResult, (long) channels * Sizeof.FLOAT * pixels, cudaMemcpyDeviceToHost);
    cudaFree(gpuResult);
    return RefIntStream.range(0, channels)
        .mapToObj(c -> RefIntStream.range(0, pixels).mapToDouble(i -> floats[c * pixels + i] * mu[0]).toArray())
        .toArray(i -> new double[i][]);
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  EigenvectorSolver_Cuda addRef() {
    return (EigenvectorSolver_Cuda) super.addRef();
  }
}
