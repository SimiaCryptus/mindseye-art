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

import com.simiacryptus.mindseye.art.photo.MultivariateFrameOfReference;
import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.RefUtil;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.jblas.Eigen;
import org.jblas.FloatMatrix;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleBinaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import static java.util.Arrays.binarySearch;

public class SparseMatrixFloat {
  @Nonnull
  public final int[] rowIndices;
  @Nonnull
  public final int[] colIndices;
  @Nonnull
  public final float[] values;
  public final int rows;
  public final int cols;

  public SparseMatrixFloat(@Nonnull int[] rowIndices, @Nonnull int[] colIndices, @Nonnull float[] values) {
    this(rowIndices, colIndices, values, Arrays.stream(rowIndices).max().getAsInt() + 1,
        Arrays.stream(colIndices).max().getAsInt() + 1);
  }

  public SparseMatrixFloat(@Nonnull int[] rowIndices, @Nonnull int[] colIndices, @Nonnull float[] values, int rows, int cols) {
    this.rows = rows;
    this.cols = cols;
    assert rowIndices.length == colIndices.length;
    assert rowIndices.length == values.length;
    this.rowIndices = rowIndices;
    this.colIndices = colIndices;
    this.values = values;
    //    assert rows == Arrays.stream(rowIndices).max().getAsInt() + 1;
    //    assert cols == Arrays.stream(colIndices).max().getAsInt() + 1;
  }

  public int[] getDenseProjection() {
    final Map<Integer, Long> rowCounts = Arrays.stream(rowIndices).mapToObj(x -> x)
        .collect(Collectors.groupingBy(x -> x, Collectors.counting()));
    final int[] activeRows = rowCounts.entrySet().stream().filter(x -> {
      boolean b = x.getValue() > 1;
      RefUtil.freeRef(x);
      return b;
    }).mapToInt(entry -> {
      Integer key = entry.getKey();
      RefUtil.freeRef(entry);
      return key;
    })
        .sorted().toArray();
    return IntStream.range(0, rows).map(x -> {
      final int newIndex = Arrays.binarySearch(activeRows, x);
      return newIndex < 0 ? 0 : newIndex;
    }).toArray();
  }

  public int getNumNonZeros() {
    return this.rowIndices.length;
  }

  @Nonnull
  public double[] getValues() {
    final double[] doubles = new double[values.length];
    for (int i = 0; i < values.length; i++) {
      doubles[i] = values[i];
    }
    return doubles;
  }

  public static double[] toDouble(@Nonnull float[] data) {
    return IntStream.range(0, data.length).mapToDouble(x -> data[x]).toArray();
  }

  @Nonnull
  public static float[] toFloat(@Nonnull double[] doubles) {
    final float[] floats = new float[doubles.length];
    for (int i = 0; i < floats.length; i++) {
      floats[i] = (float) doubles[i];
    }
    return floats;
  }

  public static long[] reorder(long[] data, @Nonnull int[] index) {
    return IntStream.range(0, index.length).mapToLong(i -> data[index[i]]).toArray();
  }

  public static int[] reorder(int[] data, @Nonnull int[] index) {
    return IntStream.range(0, index.length).map(i -> data[index[i]]).toArray();
  }

  public static double[] reorder(double[] data, @Nonnull int[] index) {
    return IntStream.range(0, index.length).mapToDouble(i -> data[index[i]]).toArray();
  }

  @Nonnull
  public static float[] reorder(float[] data, @Nonnull int[] index) {
    float[] copy = new float[10];
    int count = 0;
    for (int i = 0; i < index.length; i++) {
      float datum = data[index[i]];
      if (copy.length == count)
        copy = Arrays.copyOf(copy, count * 2);
      copy[count++] = datum;
    }
    copy = Arrays.copyOfRange(copy, 0, count);
    return copy;
  }

  public static int[] index(@Nonnull long[] data) {
    return IntStream.range(0, data.length).mapToObj(x -> x)
        .sorted(Comparator.comparingLong(i -> data[i]))
        .mapToInt(x -> x).toArray();
  }

  public static int[] index(@Nonnull int[] data) {
    return index(data, Comparator.comparingInt(i -> data[i]));
  }

  public static int[] index(@Nonnull int[] data, @Nonnull Comparator<Integer> comparator) {
    return IntStream.range(0, data.length).mapToObj(x -> x).sorted(comparator).mapToInt(x -> x).toArray();
  }

  public static int[] filterValues(@Nonnull int[] base, @Nonnull int[] filter) {
    return Arrays.stream(base).filter(i -> 0 > binarySearch(filter, i)).toArray();
  }

  public static int[] zeros(@Nonnull float[] values) {
    return zeros(values, 1e-18);
  }

  public static int[] zeros(@Nonnull float[] values, double tol) {
    return IntStream.range(0, values.length).filter(i -> Math.abs(values[i]) < tol).sorted().toArray();
  }

  @Nonnull
  public static SparseMatrixFloat identity(int size) {
    final float[] values = new float[size];
    Arrays.fill(values, 1.0f);
    return new SparseMatrixFloat(IntStream.range(0, size).toArray(), IntStream.range(0, size).toArray(), values,
        size, size).sortAndPrune();
  }

  public static int binarySearch_first(@Nonnull int[] ints, int row) {
    final int search = binarySearch(ints, row);
    if (search <= 0)
      return search;
    return IntStream.iterate(search - 1, x -> x - 1).limit(search).filter(r -> r >= 0 && ints[r] != row).findFirst()
        .orElse(-1) + 1;
  }

  public static int binarySearch_first(@Nonnull long[] ints, long row) {
    final int search = binarySearch(ints, row);
    if (search <= 0)
      return search;
    return IntStream.iterate(search - 1, x -> x - 1).limit(search).filter(r -> r >= 0 && ints[r] != row).findFirst()
        .orElse(-1) + 1;
  }

  public static int[] project(@Nonnull int[] assignments, int[] projection) {
    return Arrays.stream(assignments).map(i -> projection[i]).toArray();
  }

  @Nonnull
  public SparseMatrixFloat recalculateConnectionWeights(@Nonnull RasterTopology topology, @Nonnull Tensor content, @Nonnull int[] pixelMap,
                                                        double scale, double mixing, double minValue) {
    final MultivariateFrameOfReference globalFOR = new MultivariateFrameOfReference(() -> content.getPixelStream(), 3);
    final Map<Integer, MultivariateFrameOfReference> islandDistributions = IntStream.range(0, pixelMap.length)
        .mapToObj(x -> x).collect(Collectors.groupingBy(x -> pixelMap[x], Collectors.toList())).entrySet()
        .stream().collect(Collectors.toMap(entry -> {
          Integer key = entry.getKey();
          RefUtil.freeRef(entry);
          return key;
        }, (Map.Entry<Integer, List<Integer>> entry) -> {
          final MultivariateFrameOfReference localFOR = new MultivariateFrameOfReference(() -> entry.getValue().stream()
              .map(pixelIndex -> content.getPixel(topology.getCoordsFromIndex(pixelIndex))), 3);
          RefUtil.freeRef(entry);
          return new MultivariateFrameOfReference(globalFOR, localFOR, mixing);
        }));
    content.freeRef();
    topology.freeRef();
    return new SparseMatrixFloat(this.rowIndices, this.colIndices,
        toFloat(IntStream.range(0, this.rowIndices.length).mapToDouble(i -> {
          final MultivariateFrameOfReference a = islandDistributions.get(this.rowIndices[i]);
          final MultivariateFrameOfReference b = islandDistributions.get(this.colIndices[i]);
          assert b != null;
          assert a != null;
          double dist = a.dist(b);
          final double exp = Math.exp(-dist / scale);
          if (!Double.isFinite(exp) || exp < minValue)
            return minValue;
          return exp;
        }).toArray()), this.rows, this.cols).sortAndPrune();
  }

  @Nonnull
  public Array2DRowRealMatrix toApacheMatrix() {
    Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(rows, cols);
    for (int i = 0; i < values.length; i++) {
      matrix.setEntry(rowIndices[i], rowIndices[i], values[i]);
    }
    return matrix;
  }

  public Map<float[], Float> dense_graph_eigensys() {
    final SparseMatrixFloat sparse_W = filterDiagonal();
    final FloatMatrix matrix_W = sparse_W.toJBLAS();
    final FloatMatrix matrix_D = sparse_W.degreeMatrix().toJBLAS();
    final FloatMatrix matrix_A = matrix_D.sub(matrix_W);
    final FloatMatrix[] eigensys = Eigen.symmetricGeneralizedEigenvectors(matrix_A, matrix_D);
    final float[] realEigenvalues = eigensys[1].data;
    return IntStream.range(0, realEigenvalues.length).mapToObj(i -> i)
        .collect(Collectors.toMap(i -> eigensys[0].getColumn(i).data, i -> realEigenvalues[i]));
  }

  @Nonnull
  public FloatMatrix toJBLAS() {
    FloatMatrix matrix = new FloatMatrix(rows, cols);
    for (int i = 0; i < values.length; i++) {
      matrix.put(rowIndices[i], colIndices[i], values[i]);
    }
    return matrix;
  }

  @Nonnull
  public SparseMatrixFloat degreeMatrix() {
    return new SparseMatrixFloat(IntStream.range(0, rows).toArray(), IntStream.range(0, rows).toArray(),
        toFloat(degree()), rows, cols);
  }

  @Nonnull
  public SparseMatrixFloat project(int[] projection) {
    final int[] rowIndices = project(this.rowIndices, projection);
    final int[] colIndices = project(this.colIndices, projection);
    return new SparseMatrixFloat(rowIndices, colIndices, values, Arrays.stream(rowIndices).max().getAsInt() + 1,
        Arrays.stream(colIndices).max().getAsInt() + 1).sortAndPrune().aggregate();
  }

  public int[] activeRows() {
    return Arrays.stream(rowIndices).distinct().sorted().toArray();
  }

  @Nonnull
  public SparseMatrixFloat select(@Nonnull int[] projection) {
    final Map<Integer, Integer> reverseIndex = IntStream.range(0, projection.length).mapToObj(x -> x)
        .collect(Collectors.toMap(x -> projection[x], x -> x));
    final int[] indices = IntStream.range(0, this.rowIndices.length)
        .filter(i -> reverseIndex.containsKey(this.rowIndices[i]) && reverseIndex.containsKey(this.colIndices[i]))
        .toArray();
    final int[] rowIndices = Arrays.stream(indices).map(i -> this.rowIndices[i]).map(i -> reverseIndex.get(i))
        .toArray();
    final int[] colIndices = Arrays.stream(indices).map(i -> this.colIndices[i]).map(i -> reverseIndex.get(i))
        .toArray();
    final float[] floats = new float[indices.length];
    for (int i = 0; i < floats.length; i++) {
      floats[i] = values[indices[i]];
    }
    return new SparseMatrixFloat(rowIndices, colIndices, floats, Arrays.stream(rowIndices).max().getAsInt() + 1,
        Arrays.stream(colIndices).max().getAsInt() + 1).sortAndPrune().aggregate();
  }

  @Nonnull
  public SparseMatrixFloat sortAndPrune() {
    assert rowIndices.length == colIndices.length;
    assert rowIndices.length == values.length;
    assert Arrays.stream(rowIndices).allMatch(x -> x >= 0 && x < rows);
    assert Arrays.stream(colIndices).allMatch(x -> x >= 0 && x < cols);
    final Comparator<Integer> comparator = Comparator.comparingInt((Integer i) -> rowIndices[i])
        .thenComparingInt(i -> colIndices[i]);
    final int[] sortedAndFiltered = filterValues(index(rowIndices, comparator), zeros(values));
    return new SparseMatrixFloat(reorder(rowIndices, sortedAndFiltered), reorder(colIndices, sortedAndFiltered),
        reorder(values, sortedAndFiltered), rows, cols);
  }

  @Nonnull
  public SparseMatrixFloat copy() {
    return new SparseMatrixFloat(rowIndices, colIndices, values, rows, cols).sortAndPrune();
  }

  @Nonnull
  public SparseMatrixFloat transpose() {
    return new SparseMatrixFloat(colIndices, rowIndices, values, cols, rows).sortAndPrune();
  }

  @Nonnull
  public SparseMatrixFloat scale(double alpha) {
    final float[] values = new float[this.values.length];
    for (int i = 0; i < values.length; i++) {
      values[i] = (float) (this.values[i] * alpha);
    }
    return new SparseMatrixFloat(rowIndices, colIndices, values, rows, cols).sortAndPrune();
  }

  @Nonnull
  public SparseMatrixFloat minus(@Nonnull SparseMatrixFloat right) {
    return binaryOp(right, (a, b) -> a - b);
  }

  @Nonnull
  public SparseMatrixFloat binaryOp(@Nonnull SparseMatrixFloat right, @Nonnull DoubleBinaryOperator fn) {
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

    final long[] finalIndices = LongStream.concat(Arrays.stream(indices_left), Arrays.stream(indices_right))
        .distinct().toArray();
    long[] finalIndices_left = indices_left;
    long[] finalIndices_right = indices_right;
    return new SparseMatrixFloat(
        IntStream.range(0, finalIndices.length).mapToLong(i -> finalIndices[i]).mapToInt(elementIndex -> {
          final int i1 = (int) (elementIndex / cols);
          assert 0 <= i1 : i1;
          assert rows > i1 : i1;
          return i1;
        }).toArray(),
        IntStream.range(0, finalIndices.length).mapToLong(i -> finalIndices[i])
            .mapToInt(elementIndex -> (int) (elementIndex % cols)).toArray(),
        toFloat(IntStream.range(0, finalIndices.length).mapToLong(i -> finalIndices[i]).mapToDouble(elementIndex -> {
          final int idx_left = binarySearch_first(finalIndices_left, elementIndex);
          final int idx_right = binarySearch_first(finalIndices_right, elementIndex);
          final float value_right = idx_right < 0 || finalIndices_right[idx_right] != elementIndex ? 0
              : sortedValues_right[idx_right];
          final float value_left = idx_left < 0 || finalIndices_left[idx_left] != elementIndex ? 0
              : sortedValues_left[idx_left];
          return fn.applyAsDouble(value_left, value_right);
        }).toArray()), rows, cols).sortAndPrune();
  }

  @Nonnull
  public int[] getCols(int row) {
    final int begin = rowStart(row);
    if (begin < 0)
      return new int[]{};
    int end = getEnd(row, begin);
    return Arrays.copyOfRange(colIndices, begin, end);
  }

  public int getEnd(int row, int begin) {
    return IntStream.range(begin, rowIndices.length).filter(r -> rowIndices[r] != row).findFirst()
        .orElse(rowIndices.length);
  }

  public int rowStart(int row) {
    return binarySearch_first(this.rowIndices, row);
  }

  @Nonnull
  public float[] getVals(int row) {
    final int begin = rowStart(row);
    if (begin < 0)
      return new float[]{};
    int end = getEnd(row, begin);
    return Arrays.copyOfRange(values, begin, end);
  }

  public double getValSum(int row) {
    assert values.length == this.rowIndices.length;
    final int begin = binarySearch_first(this.rowIndices, row);
    if (begin < 0 || begin >= this.rowIndices.length)
      return 0;
    final int nnz = this.rowIndices.length;
    int end = IntStream.range(begin, nnz).filter(i -> this.rowIndices[i] != row).findFirst().orElse(nnz);
    return IntStream.range(begin, end).filter(i -> colIndices[i] != row).mapToDouble(i -> values[i]).sum();
  }

  @Nonnull
  public SparseMatrixFloat assertSymmetric() {
    assert Arrays.stream(this.activeRows()).map(x -> this.getCols(x).length).sum() == this.getNumNonZeros();
    assert Arrays.stream(this.activeRows()).map(x -> this.getVals(x).length).sum() == this.getNumNonZeros();
    assert transpose().equals(sortAndPrune(), 1e-4);
    return this;
  }

  @Nonnull
  public SparseMatrixFloat identity() {
    return SparseMatrixFloat.identity(rows);
  }

  @Nonnull
  public SparseMatrixFloat symmetricNormal() {
    double[] degree = degree();
    final float[] newValues = new float[values.length];
    for (int i = 0; i < values.length; i++) {
      newValues[i] = (float) (values[i] / Math.pow(degree[rowIndices[i]] * degree[colIndices[i]], 0.5));
    }
    return new SparseMatrixFloat(rowIndices, colIndices, newValues, rows, cols);
  }

  @Nonnull
  public SparseMatrixFloat diagonal() {
    final int[] rowIndices = new int[this.rowIndices.length];
    final int[] colIndices = new int[this.colIndices.length];
    final float[] values = new float[this.values.length];
    int sourceIndex = 0;
    int targetIndex = 0;
    while (targetIndex < values.length && sourceIndex < values.length) {
      if (this.rowIndices[sourceIndex] == this.colIndices[sourceIndex]) {
        rowIndices[targetIndex] = this.rowIndices[sourceIndex];
        colIndices[targetIndex] = this.colIndices[sourceIndex];
        values[targetIndex] = this.values[sourceIndex];
        targetIndex++;
      }
      sourceIndex++;
    }
    return new SparseMatrixFloat(Arrays.copyOf(rowIndices, targetIndex), Arrays.copyOf(colIndices, targetIndex),
        Arrays.copyOf(values, targetIndex), rows, cols);
  }

  @Nonnull
  public SparseMatrixFloat filterDiagonal() {
    final int[] rowIndices = new int[this.rowIndices.length];
    final int[] colIndices = new int[this.colIndices.length];
    final float[] values = new float[this.values.length];
    int sourceIndex = 0;
    int targetIndex = 0;
    while (targetIndex < values.length && sourceIndex < values.length) {
      if (this.rowIndices[sourceIndex] != this.colIndices[sourceIndex]) {
        rowIndices[targetIndex] = this.rowIndices[sourceIndex];
        colIndices[targetIndex] = this.colIndices[sourceIndex];
        values[targetIndex] = this.values[sourceIndex];
        targetIndex++;
      }
      sourceIndex++;
    }
    return new SparseMatrixFloat(Arrays.copyOf(rowIndices, targetIndex), Arrays.copyOf(colIndices, targetIndex),
        Arrays.copyOf(values, targetIndex), rows, cols);
  }

  @Nonnull
  public double[] degree() {
    double[] degree = new double[rows];
    for (int i = 0; i < values.length; i++) {
      degree[rowIndices[i]] += values[i];
    }
    return degree;
  }

  @Nonnull
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
    return new SparseMatrixFloat(Arrays.copyOfRange(rowIndices_copy, 0, j),
        Arrays.copyOfRange(colIndices_copy, 0, j), Arrays.copyOfRange(values_copy, 0, j), rows, cols);
  }

  private boolean equals(@Nonnull SparseMatrixFloat other, double tolerance) {
    if (rowIndices.length != other.rowIndices.length) {
      return false;
    }
    if (colIndices.length != other.colIndices.length) {
      return false;
    }
    if (values.length != other.values.length) {
      return false;
    }
    for (int i = 0; i < rowIndices.length; i++)
      if (rowIndices[i] != other.rowIndices[i]) {
        return false;
      }
    for (int i = 0; i < colIndices.length; i++)
      if (colIndices[i] != other.colIndices[i]) {
        return false;
      }
    for (int i = 0; i < values.length; i++)
      if (Math.abs(values[i] - other.values[i]) > tolerance) {
        return false;
      }
    return true;
  }

}
