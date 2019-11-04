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

import com.simiacryptus.lang.ref.LazyVal;
import com.simiacryptus.lang.ref.ReferenceCountingBase;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcusolver.cusolverSpHandle;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import org.jetbrains.annotations.NotNull;

import static jcuda.jcusolver.JCusolverSp.cusolverSpCreate;
import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

public class CudaDenseMatrix extends LazyVal<CudaDenseMatrix.GpuCopy> {

  public final SparseMatrixFloat matrix;

  public CudaDenseMatrix(SparseMatrixFloat matrix) {
    this.matrix = matrix;
  }

  @NotNull
  public static cusparseHandle newSparseHandle() {
    cusparseHandle handle = new cusparseHandle();
    cusparseCreate(handle);
    return handle;
  }

  public static cusolverSpHandle newSolverHandle() {
    cusolverSpHandle handle = new cusolverSpHandle();
    cusolverSpCreate(handle);
    return handle;
  }

  @NotNull
  public static cusparseMatDescr descriptor(int matType, int indexBase) {
    cusparseMatDescr descra = new cusparseMatDescr();
    cusparseCreateMatDescr(descra);
    cusparseSetMatType(descra, matType);
    cusparseSetMatIndexBase(descra, indexBase);
    return descra;
  }

  @NotNull
  public static Pointer toDevice(float[] values) {
    Pointer cooVal = new Pointer();
    cudaMalloc(cooVal, values.length * Sizeof.FLOAT);
    cudaMemcpy(cooVal, Pointer.to(values), values.length * Sizeof.FLOAT, cudaMemcpyHostToDevice);
    return cooVal;
  }

  @NotNull
  public static Pointer toDevice(int[] values) {
    Pointer cooRowIndex = new Pointer();
    cudaMalloc(cooRowIndex, values.length * Sizeof.INT);
    cudaMemcpy(cooRowIndex, Pointer.to(values), values.length * Sizeof.INT, cudaMemcpyHostToDevice);
    return cooRowIndex;
  }

  @Override
  @NotNull
  public CudaDenseMatrix.GpuCopy build() {
    return new GpuCopy(this);
  }

  public static final class GpuCopy extends ReferenceCountingBase {
    public final Pointer rowIndices;
    public final Pointer columnIndices;
    public final Pointer values;
    public final SparseMatrixFloat matrix;
    public final int rows;

    public GpuCopy(CudaDenseMatrix cudaCoo) {
      this.matrix = cudaCoo.matrix;
      rows = matrix.rows;
      rowIndices = toDevice(matrix.rowIndices);
      columnIndices = toDevice(matrix.colIndices);
      values = toDevice(matrix.values);
    }

    public Pointer csrRows(cusparseHandle handle) {
      Pointer csrRowPtr = new Pointer();
      cudaMalloc(csrRowPtr, (rows + 1) * Sizeof.INT);
      cusparseXcoo2csr(handle, rowIndices, matrix.rowIndices.length, rows, csrRowPtr, CUSPARSE_INDEX_BASE_ZERO);
      return csrRowPtr;
    }

    @Override
    protected void _free() {
      final GpuCopy gpuCopy = this;
      cudaFree(gpuCopy.rowIndices);
      cudaFree(gpuCopy.columnIndices);
      cudaFree(gpuCopy.values);
    }

  }
}