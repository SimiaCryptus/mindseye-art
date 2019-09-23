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
import com.simiacryptus.mindseye.art.photo.affinity.ConstAffinity;
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
import java.util.Arrays;
import java.util.stream.IntStream;

import static com.simiacryptus.mindseye.art.photo.SegmentUtil.*;
import static com.simiacryptus.mindseye.art.photo.affinity.RasterAffinity.adjust;
import static com.simiacryptus.mindseye.art.photo.affinity.RasterAffinity.degree;

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
    log.eval(() -> {
      return content.toImage();
    });
    final RasterTopology topology_simple = new SimpleRasterTopology(content.getDimensions());
    RasterTopology topology_content = new SearchRadiusTopology(content)
        .setNeighborhoodSize(6)
        .setSelfRef(true)
        .setVerbose(true);
    final Tensor contentIslands = new SmoothSolver_Cuda().solve(
        topology_content, new RelativeAffinity(content, topology_content)
            .setContrast(30)
            .setGraphPower1(2)
            .setMixing(0.5)
            .wrap((graphEdges, innerResult) -> adjust(
                graphEdges,
                innerResult,
                degree(innerResult),
                0.5)),
        1e-4).iterate(5, content);
    log.eval(() -> {
      return contentIslands.toRgbImage();
    });

    final SparseMatrixFloat graph = SmoothSolver_Cuda.laplacian(new ConstAffinity(), topology_simple).matrix;
    int[] islands = log.eval(() -> {
      return SegmentUtil.markIslands(
          topology_simple, graph, contentIslands::getPixel,
          (a, b) -> IntStream.range(0, a.length).mapToDouble(i -> a[i] - b[i]).map(x -> x * x).average().getAsDouble() < 4,
          128
      );
    });

    log.eval(() -> {
      printHistogram(islands);
      return randomColors(topology_simple, content, islands);
    });

    final SparseMatrixFloat island_graph = graph.project(islands);
    log.eval(() ->
        String.format("Rows=%d, NumNonZeros=%d", island_graph.rows, island_graph.getNumNonZeros())
    );
    final int[] islands_refined = removeTinyInclusions(islands, island_graph, 8);

    log.eval(() -> {
      final int[] refined = Arrays.stream(islands)
          .map(i -> islands_refined[i])
          .toArray();
      printHistogram(refined);
      return randomColors(topology_simple, content, refined);
    });
    final SparseMatrixFloat refined_graph = island_graph.project(islands_refined);
    log.eval(() ->
        String.format("Rows=%d, NumNonZeros=%d", refined_graph.rows, refined_graph.getNumNonZeros())
    );

    final int[] islands_joined = reduceIslands(refined_graph, 5000);
    log.eval(() -> {
      final int[] refined = Arrays.stream(islands)
          .map(i -> islands_refined[i])
          .map(i -> islands_joined[i])
          .toArray();
      printHistogram(refined);
      return randomColors(topology_simple, content, refined);
    });

  }

  @Override
  protected Class<?> getTargetClass() {
    return FastPhotoStyleTransfer.class;
  }

}
