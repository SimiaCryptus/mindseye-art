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

import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.Comparator;
import java.util.function.DoubleBinaryOperator;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

public class SparseMatrixFloat {
  public final int[] rowIndices;
  public final int[] colIndices;
  public final float[] values;
  public final int rows;
  public final int cols;

  public SparseMatrixFloat(int[] rowIndices, int[] colIndices, float[] values, int rows, int cols) {
    this.rows = rows;
    this.cols = cols;
    assert rowIndices.length == colIndices.length;
    assert rowIndices.length == values.length;
    this.rowIndices = rowIndices;
    this.colIndices = colIndices;
    this.values = values;
  }

  public static float[] toFloat(double[] doubles) {
    final float[] floats = new float[doubles.length];
    for (int i = 0; i < floats.length; i++) {
      floats[i] = (float) doubles[i];
    }
    return floats;
  }

  public static long[] reorder(long[] data, int[] index) {
    return IntStream.range(0, index.length).mapToLong(i -> data[index[i]]).toArray();
  }

  public static int[] reorder(int[] data, int[] index) {
    return IntStream.range(0, index.length).map(i -> data[index[i]]).toArray();
  }

  public static double[] reorder(double[] data, int[] index) {
    return IntStream.range(0, index.length).mapToDouble(i -> data[index[i]]).toArray();
  }

  public static float[] reorder(float[] data, int[] index) {
    float[] copy = new float[10];
    int count = 0;
    for (int i = 0; i < index.length; i++) {
      float datum = data[index[i]];
      if (copy.length == count) copy = Arrays.copyOf(copy, count * 2);
      copy[count++] = datum;
    }
    copy = Arrays.copyOfRange(copy, 0, count);
    return copy;
  }

  public static int[] index(long[] data) {
    return IntStream.range(0, data.length).mapToObj(x -> x).sorted(Comparator.comparing(i -> data[i])).mapToInt(x -> x).toArray();
  }

  public static int[] index(int[] data) {
    return index(data, Comparator.comparing(i -> data[i]));
  }

  public static int[] index(int[] data, Comparator<Integer> comparator) {
    return IntStream.range(0, data.length).mapToObj(x -> x).sorted(comparator).mapToInt(x -> x).toArray();
  }

  public static int[] filterValues(int[] base, int[] filter) {
    return Arrays.stream(base).filter(i -> 0 > Arrays.binarySearch(filter, i)).toArray();
  }

  public static int[] zeros(float[] values) {
    return zeros(values, 1e-18);
  }

  public static int[] zeros(float[] values, double tol) {
    return IntStream.range(0, values.length).filter(i -> Math.abs(values[i]) < tol).sorted().toArray();
  }

  public static SparseMatrixFloat identity(int size) {
    final float[] values = new float[size];
    Arrays.fill(values, 1.0f);
    return new SparseMatrixFloat(
        IntStream.range(0, size).toArray(),
        IntStream.range(0, size).toArray(),
        values,
        size, size).sortAndPrune();
  }

  public SparseMatrixFloat project(int[] projection) {
    final int[] rowIndices = Arrays.stream(this.rowIndices).map(i -> projection[i]).toArray();
    final int[] colIndices = Arrays.stream(this.colIndices).map(i -> projection[i]).toArray();
    return new SparseMatrixFloat(
        rowIndices,
        colIndices,
        values,
        Arrays.stream(rowIndices).max().getAsInt() + 1,
        Arrays.stream(colIndices).max().getAsInt() + 1
    ).sortAndPrune().aggregate();
  }

  protected SparseMatrixFloat aggregate() {
    assert rowIndices.length == colIndices.length;
    assert rowIndices.length == values.length;
    assert Arrays.stream(rowIndices).allMatch(x -> x >= 0 && x < rows);
    assert Arrays.stream(colIndices).allMatch(x -> x >= 0 && x < cols);
    final int[] rowIndices_copy = new int[rowIndices.length];
    final int[] colIndices_copy = new int[colIndices.length];
    final float[] values_copy = new float[values.length];
    int j = 0;
    for (int i = 0; i < rowIndices.length; i++) {
      if (i < rowIndices.length - 1 && rowIndices[i] == rowIndices[i + 1] && colIndices[i] == colIndices[i + 1]) {
        values[i + 1] += values[i];
      } else {
        rowIndices_copy[j] = rowIndices[i];
        colIndices_copy[j] = colIndices[i];
        values_copy[j] = values[i];
        j++;
      }
    }
    return new SparseMatrixFloat(
        Arrays.copyOfRange(rowIndices_copy, 0, j),
        Arrays.copyOfRange(colIndices_copy, 0, j),
        Arrays.copyOfRange(values_copy, 0, j),
        rows, cols
    );
  }

  public SparseMatrixFloat sortAndPrune() {
    assert rowIndices.length == colIndices.length;
    assert rowIndices.length == values.length;
    assert Arrays.stream(rowIndices).allMatch(x -> x >= 0 && x < rows);
    assert Arrays.stream(colIndices).allMatch(x -> x >= 0 && x < cols);
    final Comparator<Integer> comparator = Comparator.comparing((Integer i) -> rowIndices[i])
        .thenComparing((Integer i) -> colIndices[i]);
    final int[] sortedAndFiltered = filterValues(index(rowIndices, comparator), zeros(values));
    return new SparseMatrixFloat(
        reorder(rowIndices, sortedAndFiltered),
        reorder(colIndices, sortedAndFiltered),
        reorder(values, sortedAndFiltered),
        rows, cols
    );
  }

  public int getNumNonZeros() {
    return this.values.length;
  }

  public double[] getValues() {
    final double[] doubles = new double[values.length];
    for (int i = 0; i < values.length; i++) {
      doubles[i] = values[i];
    }
    return doubles;
  }

  public SparseMatrixFloat copy() {
    return new SparseMatrixFloat(rowIndices, colIndices, values, rows, cols).sortAndPrune();
  }

  public SparseMatrixFloat transpose() {
    return new SparseMatrixFloat(colIndices, rowIndices, values, cols, rows).sortAndPrune();
  }

  public SparseMatrixFloat scale(double alpha) {
    final float[] values = new float[this.values.length];
    for (int i = 0; i < values.length; i++) {
      values[i] = (float) (this.values[i] * alpha);
    }
    return new SparseMatrixFloat(rowIndices, colIndices, values, rows, cols).sortAndPrune();
  }

  public SparseMatrixFloat minus(SparseMatrixFloat right) {
    return binaryOp(right, (a, b) -> a - b);
  }

  @NotNull
  public SparseMatrixFloat binaryOp(SparseMatrixFloat right, DoubleBinaryOperator fn) {
    assert right.rows == rows : right.rows + " != " + rows;
    assert right.cols == cols : right.cols + " != " + cols;

    long[] indices_left = IntStream.range(0, rowIndices.length)
        .mapToLong(i -> (long) rowIndices[i] * cols + colIndices[i]).toArray();
    final int[] index_left = index(indices_left);
    indices_left = reorder(indices_left, index_left);
    final float[] sortedValues_left = reorder(values, index_left);

    long[] indices_right = IntStream.range(0, right.rowIndices.length)
        .mapToLong(i -> (long) right.rowIndices[i] * cols + right.colIndices[i]).toArray();
    final int[] index_right = index(indices_right);
    indices_right = reorder(indices_right, index_right);
    final float[] sortedValues_right = reorder(right.values, index_right);

    final long[] finalIndices = LongStream.concat(
        Arrays.stream(indices_left),
        Arrays.stream(indices_right)
    ).distinct().toArray();
    long[] finalIndices_left = indices_left;
    long[] finalIndices_right = indices_right;
    return new SparseMatrixFloat(
        IntStream.range(0, finalIndices.length).mapToLong(i -> finalIndices[i])
            .mapToInt(elementIndex -> {
              final int i1 = (int) (elementIndex / cols);
              assert 0 <= i1 : i1;
              assert rows > i1 : i1;
              return i1;
            }).toArray(),
        IntStream.range(0, finalIndices.length).mapToLong(i -> finalIndices[i])
            .mapToInt(elementIndex -> (int) (elementIndex % cols)).toArray(),
        toFloat(IntStream.range(0, finalIndices.length)
            .mapToLong(i -> finalIndices[i]).mapToDouble(elementIndex -> {
              final int idx_left = Arrays.binarySearch(finalIndices_left, elementIndex);
              final int idx_right = Arrays.binarySearch(finalIndices_right, elementIndex);
              final float value_right = idx_right < 0 || finalIndices_right[idx_right] != elementIndex ? 0 : sortedValues_right[idx_right];
              final float value_left = idx_left < 0 || finalIndices_left[idx_left] != elementIndex ? 0 : sortedValues_left[idx_left];
              return fn.applyAsDouble(value_left, value_right);
            }).toArray()),
        rows, cols).sortAndPrune();
  }

  public int[] getCols(int row) {
    final int begin = Arrays.binarySearch(rowIndices, row);
    if (begin < 0) return new int[]{};
    final int nnz = rowIndices.length;
    int end = IntStream.range(begin, nnz).filter(r -> rowIndices[r] != row).findFirst().orElse(nnz + 1);
    return Arrays.copyOfRange(colIndices, begin, end);
  }

  public float[] getVals(int row) {
    final int begin = Arrays.binarySearch(rowIndices, row);
    if (begin < 0) return new float[]{};
    final int nnz = rowIndices.length;
    int end = IntStream.range(begin, nnz).filter(r -> rowIndices[r] != row).findFirst().orElse(nnz + 1);
    return Arrays.copyOfRange(values, begin, end);
  }
}
