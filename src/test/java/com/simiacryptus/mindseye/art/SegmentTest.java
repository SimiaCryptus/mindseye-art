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
import com.simiacryptus.mindseye.art.photo.RegionAssembler;
import com.simiacryptus.mindseye.art.photo.SegmentUtil;
import com.simiacryptus.mindseye.art.photo.affinity.ContextAffinity;
import com.simiacryptus.mindseye.art.photo.affinity.RelativeAffinity;
import com.simiacryptus.mindseye.art.photo.cuda.SmoothSolver_Cuda;
import com.simiacryptus.mindseye.art.photo.cuda.SparseMatrixFloat;
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
import java.util.concurrent.atomic.AtomicReference;
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
  public void segment() {
    run(this::segment);
  }

  private void segment(NotebookOutput log) {
    Tensor content = contentImage();
    log.eval(() ->
        content.toImage()
    );
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
          (a, b) -> IntStream.range(0, a.length).mapToDouble(i -> a[i] - b[i]).map(x -> x * x).average().getAsDouble() < 0.5,
          128,
          pixels
      );
      pixelMap.set(islands);
      return flattenedColors;
    });
    log.eval(() -> {
      final SimpleRasterTopology topology = new SimpleRasterTopology(content.getDimensions());
      return new JoinProcess(
          content,
          topology,
          new RelativeAffinity(content, topology)
              .setContrast(50)
              .setGraphPower1(2)
              .setMixing(0.1),
          pixelMap.get());
    }).run(log);
  }

  @Override
  protected Class<?> getTargetClass() {
    return FastPhotoStyleTransfer.class;
  }

  private static class JoinProcess {
    private final Tensor content;
    private final RasterTopology topology;
    private SparseMatrixFloat graph;
    private int[] pixelMap;

    public JoinProcess(Tensor content, RasterTopology topology, ContextAffinity affinity, int[] pixelMap) {
      graph = SmoothSolver_Cuda.laplacian(affinity, topology).matrix.assertSymmetric();
      this.pixelMap = pixelMap;
      this.content = content;
      this.topology = topology;
    }

    public void run(NotebookOutput log) {
      graph = (graph.project(pixelMap).assertSymmetric());
      display(log);
      update(log.eval(() ->
          volumeEntropy(graph, pixelMap, content, topology).reduceTo(5000)
      ));
      display(log);
      update(log.eval(() ->
          volumeEntropy(graph, pixelMap, content, topology).reduceTo(500)
      ));
      display(log);
      update(log.eval(() ->
          volumeEntropy(graph, pixelMap, content, topology).reduceTo(50)
      ));
      display(log);
    }

    public void update(RegionAssembler eval) {
      graph = graph.project(eval.getProjection());
      pixelMap = eval.getPixelMap();
    }

    public void display(NotebookOutput log) {
      log.eval(() -> {
        System.out.println(String.format("Rows=%d, NumNonZeros=%d", graph.rows, graph.getNumNonZeros()));
        printHistogram(pixelMap);
        return paintWithRandomColors(topology, content, pixelMap, graph);
      });
    }
  }


}
