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

package com.simiacryptus.mindseye.art;

import com.simiacryptus.mindseye.art.photo.FastPhotoStyleTransfer;
import com.simiacryptus.mindseye.art.photo.SegmentUtil;
import com.simiacryptus.mindseye.art.photo.affinity.ContextAffinity;
import com.simiacryptus.mindseye.art.photo.affinity.RelativeAffinity;
import com.simiacryptus.mindseye.art.photo.cuda.CudaSparseMatrix;
import com.simiacryptus.mindseye.art.photo.cuda.SmoothSolver_Cuda;
import com.simiacryptus.mindseye.art.photo.cuda.SparseMatrixFloat;
import com.simiacryptus.mindseye.art.photo.topology.RadiusRasterTopology;
import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.art.photo.topology.SearchRadiusTopology;
import com.simiacryptus.mindseye.art.photo.topology.SimpleRasterTopology;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.ref.wrappers.RefSystem;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

import static com.simiacryptus.mindseye.art.photo.RegionAssembler.volumeEntropy;
import static com.simiacryptus.mindseye.art.photo.SegmentUtil.*;

public class SegmentTest extends NotebookReportBase {

  @Nonnull
  private String contentImage = "file:///C:/Users/andre/Downloads/pictures/E17-E.jpg";
  private int imageSize = 600;

  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Applications;
  }

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return FastPhotoStyleTransfer.class;
  }


  @Test
  public void segment_volumeEntropy(TestInfo testInfo) {
    run(testInfo, log -> segment_volumeEntropy(log));
  }

  @Test
  public void segment_minCut(TestInfo testInfo) {
    run(testInfo, log -> segment_minCut(log));
  }

  @Nonnull
  private Tensor contentImage() {
    return SegmentUtil.resize(ImageUtil.getTensor(contentImage), imageSize);
  }

  private void segment_volumeEntropy(@Nonnull NotebookOutput log) {
    Tensor content = contentImage();
    log.eval(() -> content.toImage());
    final int[] pixelMap = getSmoothedRegions(log, content.addRef());
    Assemble_volumeEntropy eval = log.eval(() -> {
      final SimpleRasterTopology topology = new SimpleRasterTopology(content.getDimensions());
      RelativeAffinity relativeAffinity = new RelativeAffinity(content.addRef(), topology.addRef());
      relativeAffinity.setContrast(50);
      relativeAffinity.setGraphPower1(2);
      relativeAffinity.setMixing(0.1);
      return new Assemble_volumeEntropy(content.addRef(), topology,
          relativeAffinity, pixelMap);
    });
    content.freeRef();
    eval.run(log);
    eval.freeRef();
  }

  private void segment_minCut(@Nonnull NotebookOutput log) {
    Tensor content = contentImage();
    log.eval(() -> content.toImage());
    final int[] pixelMap = getSmoothedRegions(log, content.addRef());
    Assemble_minCut minCut = log.eval(() -> {
      final RadiusRasterTopology topology = new RadiusRasterTopology(content.getDimensions(),
          RadiusRasterTopology.getRadius(1, 1), -1);
      RelativeAffinity relativeAffinity = new RelativeAffinity(content.addRef(), topology.addRef());
      relativeAffinity.setContrast(30);
      relativeAffinity.setGraphPower1(2);
      relativeAffinity.setMixing(0.5);
      return new Assemble_minCut(content.addRef(), topology,
          relativeAffinity, pixelMap);
    });
    content.freeRef();
    minCut.run(log);
    minCut.freeRef();
  }

  private int[] getSmoothedRegions(@Nonnull NotebookOutput log, @Nonnull Tensor content) {
    final AtomicReference<int[]> pixelMap = new AtomicReference(null);
    log.eval(() -> {
      SearchRadiusTopology searchRadiusTopology1 = new SearchRadiusTopology(content.addRef());
      searchRadiusTopology1.setNeighborhoodSize(6);
      SearchRadiusTopology searchRadiusTopology = searchRadiusTopology1;
      searchRadiusTopology.setSelfRef(true);
      searchRadiusTopology.setVerbose(true);
      RasterTopology topology = searchRadiusTopology.cached();
      RelativeAffinity relativeAffinity = new RelativeAffinity(content.addRef(), topology.addRef());
      relativeAffinity.setContrast(50);
      relativeAffinity.setGraphPower1(2);
      relativeAffinity.setMixing(0.1);
      final BufferedImage flattenedColors = flattenColors(content.addRef(), topology.addRef(),
          relativeAffinity, 4,
          new SmoothSolver_Cuda());
      final Tensor flattenedTensor = Tensor.fromRGB(flattenedColors);
      final int[] dimensions = topology.getDimensions();
      final int pixels = dimensions[0] * dimensions[1];
      final int[] islands = markIslands(topology, coords -> flattenedTensor.getPixel(coords), (a, b) -> RefIntStream.range(0, a.length)
          .mapToDouble(i -> a[i] - b[i]).map(x -> x * x).average().getAsDouble() < 0.2, 128, pixels);
      flattenedTensor.freeRef();
      searchRadiusTopology.freeRef();
      pixelMap.set(islands);
      return flattenedColors;
    });
    content.freeRef();
    return pixelMap.get();
  }

  private static class Assemble_minCut extends ReferenceCountingBase {
    private final Tensor content;
    @Nonnull
    private final RasterTopology topology;
    private SparseMatrixFloat graph;
    private int[] pixelMap;

    public Assemble_minCut(Tensor content, @Nonnull RasterTopology topology, @Nonnull ContextAffinity affinity, int[] pixelMap) {
      this.pixelMap = pixelMap;
      CudaSparseMatrix laplacian = SmoothSolver_Cuda.laplacian(affinity, topology.addRef());
      this.graph = laplacian.matrix.assertSymmetric().project(this.pixelMap);
      laplacian.freeRef();
      this.content = content;
      this.topology = topology;
    }

    public void run(@Nonnull NotebookOutput log) {
      display(log);
      update(volumeEntropy(graph, pixelMap, content.addRef(), topology.addRef()).reduceTo(10000).getProjection());
      update(graph.getDenseProjection());
      display(log);

      final double scale = 5e-3;
      this.graph = this.graph.recalculateConnectionWeights(topology.addRef(), content.addRef(), pixelMap, scale, 0.5, 1e-9);
      update(volumeEntropy(graph, pixelMap, content.addRef(), topology.addRef()).reduceTo(1000).getProjection());
      update(graph.getDenseProjection());
      display(log);
      this.graph = this.graph.recalculateConnectionWeights(topology.addRef(), content.addRef(), pixelMap, scale, 0.5, 1e-9);
      update(graph.getDenseProjection());
      // Arrays.stream(SparseMatrixFloat.toDouble(this.graph.values)).mapToObj(x->x).sorted(Comparator.comparing(x->-x)).mapToDouble(x->x).toArray()
      // Arrays.stream(SparseMatrixFloat.toDouble(this.graph.values)).sorted().toArray()
      final Map<float[], Float> eigensystem = log.eval(() -> this.graph.dense_graph_eigensys());
      log.run(() -> {
        RefSystem.out.println("Sorted Eigenvalues: "
            + RefArrays.toString(eigensystem.values().stream().mapToDouble(aFloat -> aFloat.doubleValue()).toArray()));
      });
      final int sampleEigenvectors = 20;
      log.h2("Smallest Eigenvectors");
      int index = 0;
      for (Map.Entry<float[], Float> tuple : eigensystem.entrySet().stream()
          .sorted(Comparator.comparingDouble(x -> {
            Float value = x.getValue();
            RefUtil.freeRef(x);
            return value;
          }))
          .limit(sampleEigenvectors)
          .collect(Collectors.toList())) {
        log.h3("Eigenvector " + index++);
        log.eval(() -> {
          return tuple.getValue();
        });
        final double[] vector = SparseMatrixFloat.toDouble(tuple.getKey());
        updateAndDisplay(log, Arrays.stream(vector).mapToInt(x -> x < 0 ? 0 : x == 0 ? 1 : 2).toArray());
        final double median = log.eval(() -> {
          return Arrays.stream(vector).sorted().toArray()[vector.length / 2];
        });
        updateAndDisplay(log, Arrays.stream(vector).mapToInt(x -> x < median ? 0 : x == median ? 1 : 2).toArray());
      }

      log.h2("Largest Eigenvectors");
      index = 0;
      for (Map.Entry<float[], Float> tuple : eigensystem.entrySet().stream()
          .sorted(Comparator.comparing(entry -> {
            float v = -entry.getValue();
            RefUtil.freeRef(entry);
            return v;
          }))
          .limit(sampleEigenvectors)
          .collect(Collectors.toList())) {
        log.h3("Eigenvector " + index++);
        log.eval(() -> tuple.getValue());
        final double[] vector = SparseMatrixFloat.toDouble(tuple.getKey());
        updateAndDisplay(log, RefArrays.stream(vector).mapToInt(x -> x < 0 ? 0 : x == 0 ? 1 : 2).toArray());
        final double median = log.eval(() -> RefArrays.stream(vector).sorted().toArray()[vector.length / 2]);
        updateAndDisplay(log, RefArrays.stream(vector).mapToInt(x -> x < median ? 0 : x == median ? 1 : 2).toArray());
      }

      log.h2("Mincut Eigenvector");
      update(log.eval(() -> {
        //final double[] secondSmallestEigenvector = eigenDecomposition.getEigenvector(sortedIndexes[1]).toArray();
        final Map.Entry<float[], Float> secondLowest = eigensystem.entrySet().stream()
            .sorted(Comparator.comparing(entry -> {
              Float value = entry.getValue();
              RefUtil.freeRef(entry);
              return value;
            }))
            .limit(sampleEigenvectors)
            .collect(Collectors.toList())
            .get(1);
        RefSystem.out.println("Second Smallest Eigenvector " + RefArrays.toString(secondLowest.getKey()));
        int[] ints = Arrays.stream(SparseMatrixFloat.toDouble(secondLowest.getKey()))
            .mapToInt(x -> x < 0 ? 0 : 1).toArray();
        RefUtil.freeRef(secondLowest);
        return ints;
      }));
      display(log);
    }

    public void display(@Nonnull NotebookOutput log) {
      display(log, pixelMap, graph);
    }

    public void updateAndDisplay(@Nonnull NotebookOutput log, int[] projection) {
      display(log, SparseMatrixFloat.project(pixelMap, projection), graph.project(projection));
    }

    public void update(int[] projection) {
      graph = graph.project(projection);
      pixelMap = SparseMatrixFloat.project(pixelMap, projection);
    }

    public void display(@Nonnull NotebookOutput log, @Nonnull int[] pixelMap, @Nonnull SparseMatrixFloat graph) {
      log.eval(() -> {
        RefSystem.out
            .println(RefString.format("Rows=%d, NumNonZeros=%d", graph.rows, graph.getNumNonZeros()));
        printHistogram(pixelMap);
        return paintWithRandomColors(topology.addRef(), pixelMap, graph);
      });
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      content.freeRef();
      topology.freeRef();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Assemble_minCut addRef() {
      return (Assemble_minCut) super.addRef();
    }
  }

  private static class Assemble_volumeEntropy extends ReferenceCountingBase {
    private final Tensor content;
    @Nonnull
    private final RasterTopology topology;
    private SparseMatrixFloat graph;
    private int[] pixelMap;

    public Assemble_volumeEntropy(Tensor content, @Nonnull RasterTopology topology, @Nonnull ContextAffinity affinity, int[] pixelMap) {
      this.pixelMap = pixelMap;
      CudaSparseMatrix laplacian = SmoothSolver_Cuda.laplacian(affinity, topology.addRef());
      this.graph = laplacian.matrix.assertSymmetric().project(this.pixelMap);
      laplacian.freeRef();
      this.content = content;
      this.topology = topology;
    }

    public void run(@Nonnull NotebookOutput log) {
      display(log);
      update(log.eval(() -> {
        return volumeEntropy(graph, pixelMap, content.addRef(), topology.addRef()).reduceTo(5000).getProjection();
      }));
      display(log);
      update(log.eval(() -> {
        return volumeEntropy(graph, pixelMap, content.addRef(), topology.addRef()).reduceTo(500).getProjection();
      }));
      display(log);
      update(log.eval(() -> {
        return volumeEntropy(graph, pixelMap, content.addRef(), topology.addRef()).reduceTo(50).getProjection();
      }));
      display(log);
    }

    public void update(int[] projection) {
      graph = graph.project(projection);
      pixelMap = SparseMatrixFloat.project(pixelMap, projection);
    }

    public void display(@Nonnull NotebookOutput log) {
      log.eval(() -> {
        RefSystem.out
            .println(RefString.format("Rows=%d, NumNonZeros=%d", graph.rows, graph.getNumNonZeros()));
        printHistogram(pixelMap);
        return paintWithRandomColors(topology.addRef(), pixelMap, graph);
      });
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      content.freeRef();
      topology.freeRef();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Assemble_volumeEntropy addRef() {
      return (Assemble_volumeEntropy) super.addRef();
    }
  }

}
