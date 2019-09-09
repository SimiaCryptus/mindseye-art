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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.ImgCropLayer;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.interfaces.linsol.LinearSolverSparse;
import org.ejml.simple.SimpleMatrix;
import org.ejml.sparse.FillReducing;
import org.ejml.sparse.csc.factory.LinearSolverFactory_DSCC;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.List;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class PixelGraph {

  private final Tensor content;
  private final int[] dimensions;
  private final int max_neighborhood_size = 9;

  public PixelGraph(Tensor content) {
    this.content = content;
    this.dimensions = content.getDimensions();
  }

  @NotNull
  public static Tensor crop(Tensor content, Tensor vector) {
    final int[] dimensions = content.getDimensions();
    final ImgCropLayer imgCropLayer = new ImgCropLayer(dimensions[0], dimensions[1]);
    final Tensor tensor = imgCropLayer.eval(vector).getDataAndFree().getAndFree(0);
    imgCropLayer.freeRef();
    return tensor;
  }

  /**
   * Implemented PhotoSmooth algorithm detailed in:
   * A Closed-form Solution to Photorealistic Image Stylization
   * https://arxiv.org/pdf/1802.06474.pdf
   */
  public UnaryOperator<Tensor> smoothingTransform(double lambda, double epsilon) {
    return wrap(solve(affinity(epsilon), lambda));
  }

  public UnaryOperator<SimpleMatrix> solve(DMatrixSparseCSC affinity, double lambda) {
    final double alpha = 1.0 / (1.0 + lambda);
    final LinearSolverSparse<DMatrixSparseCSC, DMatrixRMaj> solver = LinearSolverFactory_DSCC.lu(FillReducing.NONE);
    solver.setA(laplacian(affinity, alpha));
    return img -> {
      SimpleMatrix smoothed = new SimpleMatrix(img.numRows(), img.numCols());
      solver.solve(img.getDDRM(), smoothed.getDDRM());
      return smoothed.scale(1 - alpha);
    };
  }

  public DMatrixSparseCSC laplacian(DMatrixSparseCSC affinity, double alpha) {
    final SimpleMatrix identity = SimpleMatrix.identity(dimensions[0] * dimensions[1], DMatrixSparseCSC.class);
    return identity.minus(new SimpleMatrix(affinity).scale(alpha)).getDSCC();
  }

  @NotNull
  public UnaryOperator<Tensor> wrap(UnaryOperator<SimpleMatrix> solver) {
    return tensor -> {
      if (!Arrays.equals(dimensions, tensor.getDimensions()))
        throw new IllegalArgumentException(Arrays.toString(dimensions) + " != " + Arrays.toString(tensor.getDimensions()));
      final SimpleMatrix imageMatrix = new SimpleMatrix(dimensions[0] * dimensions[1], dimensions[2]);
      for (int x = 0; x < dimensions[0]; x++) {
        for (int y = 0; y < dimensions[1]; y++) {
          for (int c = 0; c < dimensions[2]; c++) {
            imageMatrix.set(getIndexFromCoords(x, y), c, tensor.get(x, y, c));
          }
        }
      }
      SimpleMatrix smoothed = solver.apply(imageMatrix);
      return tensor.mapCoords(coordinate -> {
        final int[] c = coordinate.getCoords();
        return Math.min(Math.max(smoothed.get(getIndexFromCoords(c[0], c[1]), c[2]), 0), 255);
      });
    };
  }

  /**
   * Implements Matting Affinity
   * <p>
   * See Also: A Closed Form Solution to Natural Image Matting
   * http://cs.brown.edu/courses/cs129/results/final/valayshah/Matting-Levin-Lischinski-Weiss-CVPR06.pdf
   */
  protected double affinity(int x1, int y1, int x2, int y2, double epsilon) {
    //assert x1 != x2 || y1 != y2;
    final int bands = dimensions[2];
    final int scale = 256;
    final List<double[]> neighborhood = neighborPixels(x1, y1, x2, y2, 3)
        .stream().map(a -> Arrays.stream(a).map(v -> v / scale).toArray())
        .collect(Collectors.toList());
    final SimpleMatrix means = new SimpleMatrix(bands, 1);
    IntStream.range(0, bands).forEach(c -> means.set(c, 0, neighborhood.stream().mapToDouble(p -> p[c]).average().getAsDouble()));
    final SimpleMatrix cov = new SimpleMatrix(bands, bands);
    IntStream.range(0, bands).forEach(c1 ->
        IntStream.range(0, bands).forEach(c2 -> {
          final double mean1 = means.get(c1);
          final double mean2 = means.get(c2);
          cov.set(c1, c2, neighborhood.stream().mapToDouble(p -> (p[c1] - mean1) * (p[c2] - mean2)).average().getAsDouble());
        })
    );
    double size = neighborhood.size();
    final SimpleMatrix invert = cov.plus(SimpleMatrix.identity(bands).scale(epsilon / size)).invert();
    final SimpleMatrix i1 = new SimpleMatrix(bands, 1, false, IntStream.range(0, bands).mapToDouble(c -> (content.get(x1, y1, c) / scale) - means.get(c)).toArray());
    final SimpleMatrix i2 = new SimpleMatrix(bands, 1, false, IntStream.range(0, bands).mapToDouble(c -> (content.get(x2, y2, c) / scale) - means.get(c)).toArray());
    final double v = (1 + i1.dot(invert.mult(i2))) / size;
    return Math.max(0, v);
  }

  protected List<double[]> neighborPixels(int x1, int y1, int x2, int y2, int minSize) {
    final int bands = dimensions[2];
    int min_x = Math.min(x1, x2);
    int max_x = Math.max(x1, x2);
    while ((max_x - min_x) < minSize - 1) {
      if (min_x > 0) min_x--;
      if ((max_x - min_x) < minSize - 1)
        if (max_x < dimensions[0] - 1) max_x++;
        else if (min_x <= 0) break;
    }
    return IntStream.range(min_x, max_x + 1).mapToObj(x -> x).flatMap(x -> {
      int min_y = Math.min(y1, y2);
      int max_y = Math.max(y1, y2);
      while ((max_y - min_y) < minSize - 1) {
        if (min_y > 0) min_y--;
        if ((max_y - min_y) < minSize - 1)
          if (max_y < dimensions[1] - 1) max_y++;
          else if (min_y <= 0) break;
      }
      return IntStream.range(min_y, max_y + 1).mapToObj(y -> y).map(y ->
          IntStream.range(0, bands).mapToDouble(c -> content.get(x, y, c)).toArray());
    }).collect(Collectors.toList());
  }

  protected List<int[]> connectivity() {
    return connectivity(1);
  }

  protected List<int[]> connectivity(int pow) {
    assert pow > 0;
    if (1 == pow) {
      final ThreadLocal<int[]> neighbors = ThreadLocal.withInitial(() -> new int[max_neighborhood_size]);
      return IntStream.range(0, dimensions[0] * dimensions[1]).parallel().mapToObj(i -> {
        final int[] original = neighbors.get();
        return Arrays.copyOf(original, getNeighbors(getCoordsFromIndex(i), original));
      }).collect(Collectors.toList());
    } else {
      final List<int[]> prev = connectivity(pow - 1);
      return IntStream.range(0, prev.size()).mapToObj(j ->
          Arrays.stream(prev.get(j))
              .flatMap(i -> Arrays.stream(prev.get(i)))
              .filter(i -> i != j)
              .distinct().toArray()
      ).collect(Collectors.toList());
    }
  }

  private int getNeighbors(int[] coords, int[] neighbors) {
    int[] dimensions = content.getDimensions();
    int neighborCount = 0;
    final int x = coords[0];
    final int y = coords[1];
    final int w = dimensions[0];
    final int h = dimensions[1];
    if (y > 0) {
      if (x > 0) {
        neighbors[neighborCount++] = getIndexFromCoords(x - 1, y - 1);
      }
      if (x < w - 1) {
        neighbors[neighborCount++] = getIndexFromCoords(x + 1, y - 1);
      }
      neighbors[neighborCount++] = getIndexFromCoords(x, y - 1);
    }
    if (y < h - 1) {
      if (x > 0) {
        neighbors[neighborCount++] = getIndexFromCoords(x - 1, y + 1);
      }
      if (x < w - 1) {
        neighbors[neighborCount++] = getIndexFromCoords(x + 1, y + 1);
      }
      neighbors[neighborCount++] = getIndexFromCoords(x, y + 1);
    }
    if (x > 0) {
      neighbors[neighborCount++] = getIndexFromCoords(x - 1, y);
    }
    if (x < w - 1) {
      neighbors[neighborCount++] = getIndexFromCoords(x + 1, y);
    }
    //neighbors[neighborCount++] = getIndexFromCoords(x, y);
    return neighborCount;
  }

  @NotNull
  public DMatrixSparseCSC affinity(double epsilon) {
    final int pixels = dimensions[0] * dimensions[1];
    final List<int[]> graphEdges = connectivity();
    // Calclulate affinity matrix
    final List<double[]> affinityList = IntStream.range(0, pixels).parallel().mapToObj(i -> {
          final int[] ci = getCoordsFromIndex(i);
          return Arrays.stream(graphEdges.get(i)).mapToDouble(j -> {
            final int[] cj = getCoordsFromIndex(j);
            return affinity(ci[0], ci[1], cj[0], cj[1], epsilon);
          }).toArray();
        }
    ).collect(Collectors.toList());
    final DMatrixSparseCSC adjacency = new DMatrixSparseCSC(pixels, pixels);
    for (int i = 0; i < pixels; i++) {
      double[] affinities = affinityList.get(i);
      final int[] edges = graphEdges.get(i);
      assert affinities.length == edges.length;
      for (int j = 0; j < edges.length; j++) {
        final double affinity = affinities[j];
        if (affinity > 0) adjacency.set(i, edges[j], affinity);
      }
    }
    final double[] degree = affinityList.stream().mapToDouble(x -> Arrays.stream(x).sum()).toArray();
    // Calclulate normalized laplacian
    final DMatrixSparseCSC rescaled = new DMatrixSparseCSC(pixels, pixels);
    for (int i = 0; i < pixels; i++) {
      final double deg_i = degree[i];
      if (deg_i == 0) continue;
      for (int j : graphEdges.get(i)) {
        //assert i != j;
        if (i > j) continue;
        final double deg_j = degree[j];
        if (deg_j == 0) continue;
        final double adj = adjacency.get(i, j);
        if (adj == 0) continue;
        final double val = adj / Math.sqrt(deg_j * deg_i);
        rescaled.set(i, j, val);
        rescaled.set(j, i, val);
      }
    }
    return rescaled;
  }

  protected int getIndexFromCoords(int x, int y) {
    return x + dimensions[0] * y;
  }

  protected int[] getCoordsFromIndex(int i) {
    final int x = i % dimensions[0];
    final int y = (i - x) / dimensions[0];
    return new int[]{x, y};
  }


}
