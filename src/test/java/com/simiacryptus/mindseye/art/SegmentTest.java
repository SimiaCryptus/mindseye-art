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
import org.jetbrains.annotations.NotNull;
import org.junit.Test;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.simiacryptus.mindseye.art.photo.RegionAssembler.volumeEntropy;
import static com.simiacryptus.mindseye.art.photo.SegmentUtil.*;

public class SegmentTest extends NotebookReportBase {

  private String contentImage = "file:///C:/Users/andre/Downloads/pictures/E17-E.jpg";
  private int imageSize = 600;

  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Applications;
  }

  @NotNull
  private Tensor contentImage() {
    return SegmentUtil.resize(ImageUtil.getTensor(contentImage), imageSize);
  }

  @Test
  public void segment_volumeEntropy() {
    run(this::segment_volumeEntropy);
  }

  private void segment_volumeEntropy(NotebookOutput log) {
    Tensor content = contentImage();
    log.eval(() ->
        content.toImage()
    );
    final int[] pixelMap = getSmoothedRegions(log, content);
    log.eval(() -> {
      final SimpleRasterTopology topology = new SimpleRasterTopology(content.getDimensions());
      return new Assemble_volumeEntropy(
          content,
          topology,
          new RelativeAffinity(content, topology)
              .setContrast(50)
              .setGraphPower1(2)
              .setMixing(0.1),
          pixelMap);
    }).run(log);
  }

  @Test
  public void segment_minCut() {
    run(this::segment_minCut);
  }

  private void segment_minCut(NotebookOutput log) {
    Tensor content = contentImage();
    log.eval(() ->
        content.toImage()
    );
    final int[] pixelMap = getSmoothedRegions(log, content);
    log.eval(() -> {
      final RadiusRasterTopology topology = new RadiusRasterTopology(content.getDimensions(), RadiusRasterTopology.getRadius(1, 1), -1);
      return new Assemble_minCut(
          content,
          topology,
          new RelativeAffinity(content, topology)
              .setContrast(30)
              .setGraphPower1(2)
              .setMixing(0.5),
          pixelMap);
    }).run(log);
  }

  private int[] getSmoothedRegions(NotebookOutput log, Tensor content) {
    final AtomicReference<int[]> pixelMap = new AtomicReference(null);
    log.eval(() -> {
      RasterTopology topology = new SearchRadiusTopology(content)
          .setNeighborhoodSize(6).setSelfRef(true).setVerbose(true).cached();
      final BufferedImage flattenedColors = flattenColors(content, topology,
          new RelativeAffinity(content, topology)
              .setContrast(50)
              .setGraphPower1(2)
              .setMixing(0.1), 4, new SmoothSolver_Cuda());
      final Tensor flattenedTensor = Tensor.fromRGB(flattenedColors);
      final int[] dimensions = topology.getDimensions();
      final int pixels = dimensions[0] * dimensions[1];
      final int[] islands = markIslands(
          topology, flattenedTensor::getPixel,
          (a, b) -> IntStream.range(0, a.length).mapToDouble(i -> a[i] - b[i]).map(x -> x * x).average().getAsDouble() < 0.2,
          128,
          pixels
      );
      pixelMap.set(islands);
      return flattenedColors;
    });
    return pixelMap.get();
  }

  @Override
  protected Class<?> getTargetClass() {
    return FastPhotoStyleTransfer.class;
  }

  private static class Assemble_minCut {
    private final Tensor content;
    private final RasterTopology topology;
    private SparseMatrixFloat graph;
    private int[] pixelMap;

    public Assemble_minCut(Tensor content, RasterTopology topology, ContextAffinity affinity, int[] pixelMap) {
      this.pixelMap = pixelMap;
      this.graph = SmoothSolver_Cuda.laplacian(affinity, topology).matrix.assertSymmetric().project(this.pixelMap);
      this.content = content;
      this.topology = topology;
    }

    public void run(NotebookOutput log) {
      display(log);
      update(volumeEntropy(graph, pixelMap, content, topology).reduceTo(10000).getProjection());
      update(graph.getDenseProjection());
      display(log);

      final double scale = 5e-3;
      this.graph = this.graph.recalculateConnectionWeights(topology, content, pixelMap, scale, 0.5, 1e-9);
      update(volumeEntropy(graph, pixelMap, content, topology).reduceTo(1000).getProjection());
      update(graph.getDenseProjection());
      display(log);
      this.graph = this.graph.recalculateConnectionWeights(topology, content, pixelMap, scale, 0.5, 1e-9);
      update(graph.getDenseProjection());
      // Arrays.stream(SparseMatrixFloat.toDouble(this.graph.values)).mapToObj(x->x).sorted(Comparator.comparing(x->-x)).mapToDouble(x->x).toArray()
      // Arrays.stream(SparseMatrixFloat.toDouble(this.graph.values)).sorted().toArray()
      final Map<float[], Float> eigensystem = log.eval(() ->
          this.graph.dense_graph_eigensys()
      );
      log.run(() -> {
        System.out.println("Sorted Eigenvalues: " + Arrays.toString(eigensystem.values().stream().mapToDouble(Float::doubleValue).toArray()));
      });
      final int sampleEigenvectors = 20;
      log.h2("Smallest Eigenvectors");
      int index = 0;
      for (Map.Entry<float[], Float> tuple : eigensystem.entrySet().stream().sorted(Comparator.comparing(x -> x.getValue())).limit(sampleEigenvectors).collect(Collectors.toList())) {
        log.h3("Eigenvector " + index++);
        log.eval(() ->
            tuple.getValue()
        );
        final double[] vector = SparseMatrixFloat.toDouble(tuple.getKey());
        updateAndDisplay(log, Arrays.stream(vector).mapToInt(x -> x < 0 ? 0 : x == 0 ? 1 : 2).toArray());
        final double median = log.eval(() ->
            Arrays.stream(vector).sorted().toArray()[vector.length / 2]
        );
        updateAndDisplay(log, Arrays.stream(vector).mapToInt(x -> x < median ? 0 : x == median ? 1 : 2).toArray());
      }

      log.h2("Largest Eigenvectors");
      index = 0;
      for (Map.Entry<float[], Float> tuple : eigensystem.entrySet().stream().sorted(Comparator.comparing(x -> -x.getValue())).limit(sampleEigenvectors).collect(Collectors.toList())) {
        log.h3("Eigenvector " + index++);
        log.eval(() ->
            tuple.getValue()
        );
        final double[] vector = SparseMatrixFloat.toDouble(tuple.getKey());
        updateAndDisplay(log, Arrays.stream(vector).mapToInt(x -> x < 0 ? 0 : x == 0 ? 1 : 2).toArray());
        final double median = log.eval(() ->
            Arrays.stream(vector).sorted().toArray()[vector.length / 2]
        );
        updateAndDisplay(log, Arrays.stream(vector).mapToInt(x -> x < median ? 0 : x == median ? 1 : 2).toArray());
      }

      log.h2("Mincut Eigenvector");
      update(log.eval(() -> {
        //final double[] secondSmallestEigenvector = eigenDecomposition.getEigenvector(sortedIndexes[1]).toArray();
        final Map.Entry<float[], Float> secondLowest = eigensystem.entrySet().stream().sorted(Comparator.comparing(x -> x.getValue())).limit(sampleEigenvectors).collect(Collectors.toList()).get(1);
        System.out.println("Second Smallest Eigenvector " + Arrays.toString(secondLowest.getKey()));
        return Arrays.stream(SparseMatrixFloat.toDouble(secondLowest.getKey())).mapToInt(x -> x < 0 ? 0 : 1).toArray();
      }));
      display(log);
    }

    public void display(NotebookOutput log) {
      display(log, pixelMap, graph);
    }

    public void updateAndDisplay(NotebookOutput log, int[] projection) {
      display(log, SparseMatrixFloat.project(pixelMap, projection), graph.project(projection));
    }

    public void update(int[] projection) {
      graph = graph.project(projection);
      pixelMap = SparseMatrixFloat.project(pixelMap, projection);
    }

    public void display(NotebookOutput log, int[] pixelMap, SparseMatrixFloat graph) {
      log.eval(() -> {
        System.out.println(String.format("Rows=%d, NumNonZeros=%d", graph.rows, graph.getNumNonZeros()));
        printHistogram(pixelMap);
        return paintWithRandomColors(topology, pixelMap, graph);
      });
    }
  }

  private static class Assemble_volumeEntropy {
    private final Tensor content;
    private final RasterTopology topology;
    private SparseMatrixFloat graph;
    private int[] pixelMap;

    public Assemble_volumeEntropy(Tensor content, RasterTopology topology, ContextAffinity affinity, int[] pixelMap) {
      this.pixelMap = pixelMap;
      this.graph = SmoothSolver_Cuda.laplacian(affinity, topology).matrix.assertSymmetric().project(this.pixelMap);
      this.content = content;
      this.topology = topology;
    }

    public void run(NotebookOutput log) {
      display(log);
      update(log.eval(() ->
          volumeEntropy(graph, pixelMap, content, topology).reduceTo(5000).getProjection()
      ));
      display(log);
      update(log.eval(() ->
          volumeEntropy(graph, pixelMap, content, topology).reduceTo(500).getProjection()
      ));
      display(log);
      update(log.eval(() ->
          volumeEntropy(graph, pixelMap, content, topology).reduceTo(50).getProjection()
      ));
      display(log);
    }

    public void update(int[] projection) {
      graph = graph.project(projection);
      pixelMap = SparseMatrixFloat.project(pixelMap, projection);
    }

    public void display(NotebookOutput log) {
      log.eval(() -> {
        System.out.println(String.format("Rows=%d, NumNonZeros=%d", graph.rows, graph.getNumNonZeros()));
        printHistogram(pixelMap);
        return paintWithRandomColors(topology, pixelMap, graph);
      });
    }
  }


}
