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

import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.Comparator;
import java.util.function.DoubleBinaryOperator;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

public class SparseMatrix {
  public final int[] rowIndices;
  public final int[] colIndices;
  public final float[] values;
  public final int rows;
  public final int cols;


  public SparseMatrix(int[] rowIndices, int[] colIndices, float[] values, int rows, int cols) {
    this.rows = rows;
    this.cols = cols;
    assert rowIndices.length == colIndices.length;
    assert rowIndices.length == values.length;
    assert Arrays.stream(rowIndices).allMatch(x -> x >= 0 && x < rows);
    assert Arrays.stream(colIndices).allMatch(x -> x >= 0 && x < cols);
    final int[] sortedAndFiltered = subtractIndices(index(rowIndices), zeros(values));
    this.rowIndices = reorder(rowIndices, sortedAndFiltered);
    this.colIndices = reorder(colIndices, sortedAndFiltered);
    this.values = reorder(values, sortedAndFiltered);
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
    return IntStream.range(0, data.length).mapToObj(x -> x).sorted(Comparator.comparing(i -> data[i])).mapToInt(x -> x).toArray();
  }

  public static int[] subtractIndices(int[] left, int[] right) {
    return Arrays.stream(left).filter(i -> 0 > Arrays.binarySearch(right, i)).toArray();
  }

  public static int[] zeros(float[] values) {
    return zeros(values, 1e-18);
  }

  public static int[] zeros(float[] values, double tol) {
    return IntStream.range(0, values.length).filter(i -> Math.abs(values[i]) < tol).sorted().toArray();
  }

  public static SparseMatrix identity(int size) {
    final float[] values = new float[size];
    Arrays.fill(values, 1.0f);
    return new SparseMatrix(
        IntStream.range(0, size).toArray(),
        IntStream.range(0, size).toArray(),
        values,
        size, size);
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

  public SparseMatrix copy() {
    return new SparseMatrix(rowIndices, colIndices, values, rows, cols);
  }

  public SparseMatrix transpose() {
    return new SparseMatrix(colIndices, rowIndices, values, cols, rows);
  }

  public SparseMatrix scale(double alpha) {
    final float[] values = new float[this.values.length];
    for (int i = 0; i < values.length; i++) {
      values[i] = (float) (this.values[i] * alpha);
    }
    return new SparseMatrix(rowIndices, colIndices, values, rows, cols);
  }

  public SparseMatrix minus(SparseMatrix right) {
    return binaryOp(right, (a, b) -> a - b);
  }

  @NotNull
  public SparseMatrix binaryOp(SparseMatrix right, DoubleBinaryOperator fn) {
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
    return new SparseMatrix(
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
        rows, cols);
  }
}
