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

public class RasterSolver_EJML implements RasterSolver {

  public static UnaryOperator<SimpleMatrix> solve(DMatrixSparseCSC affinity, double lambda) {
    final double alpha = 1.0 / (1.0 + lambda);
    final LinearSolverSparse<DMatrixSparseCSC, DMatrixRMaj> solver = LinearSolverFactory_DSCC.cholesky(FillReducing.NONE);
    final SimpleMatrix identity = SimpleMatrix.identity(affinity.numRows, DMatrixSparseCSC.class);
    solver.setA(identity.scale(1).minus(new SimpleMatrix(affinity).scale(alpha)).getDSCC());
    return img -> {
      SimpleMatrix smoothed = new SimpleMatrix(img.numRows(), img.numCols());
      solver.solve(img.getDDRM(), smoothed.getDDRM());
      return smoothed.scale(1 - alpha);
    };
  }

  @NotNull
  public static DMatrixSparseCSC laplacian(RasterAffinity affinity) {
    return laplacian(affinity.getTopology().connectivity(), affinity.affinityList(affinity.getTopology().connectivity()));
  }

  @NotNull
  public static UnaryOperator<Tensor> wrap(UnaryOperator<SimpleMatrix> solver, RasterTopology topology) {
    final int[] dimensions = topology.getDimensions();
    return tensor -> {
      if (!Arrays.equals(dimensions, tensor.getDimensions()))
        throw new IllegalArgumentException(Arrays.toString(dimensions) + " != " + Arrays.toString(tensor.getDimensions()));
      final SimpleMatrix imageMatrix = new SimpleMatrix(dimensions[0] * dimensions[1], dimensions[2]);
      for (int x = 0; x < dimensions[0]; x++) {
        for (int y = 0; y < dimensions[1]; y++) {
          for (int c = 0; c < dimensions[2]; c++) {
            imageMatrix.set(topology.getIndexFromCoords(x, y), c, tensor.get(x, y, c));
          }
        }
      }
      SimpleMatrix smoothed = solver.apply(imageMatrix);
      return tensor.mapCoords(coordinate -> {
        final int[] c = coordinate.getCoords();
        return Math.min(Math.max(smoothed.get(topology.getIndexFromCoords(c[0], c[1]), c[2]), 0), 255);
      });
    };
  }

  public static @NotNull DMatrixSparseCSC laplacian(List<int[]> graphEdges, List<double[]> affinityList) {
    final int pixels = graphEdges.size();
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

  @NotNull
  public RefOperator<Tensor> smoothingTransform(double lambda, RasterAffinity affinity) {
    return RefOperator.wrap(wrap(solve(laplacian(affinity), lambda), affinity.getTopology()));
  }


}
