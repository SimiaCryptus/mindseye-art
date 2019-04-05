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

import com.simiacryptus.lang.ref.ReferenceCountingBase;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.cudnn.ImgTileSelectLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.NullNotebookOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * The type Tiled trainable.
 */
public abstract class TiledTrainable extends ReferenceCountingBase implements Trainable {

  private static final Logger logger = LoggerFactory.getLogger(TiledTrainable.class);

  private final Tensor canvas;
  private final Layer filter;
  private final ImgTileSelectLayer[] selectors;
  private final PipelineNetwork[] networks;
  private final UUID canvasId = UUID.randomUUID();
  AtomicInteger count = new AtomicInteger();
  private NotebookOutput log = new NullNotebookOutput();
  private boolean verbose = false;

  public TiledTrainable(Tensor canvas, int tileSize, int padding) {
    this(canvas, new PipelineNetwork(1), tileSize, padding);
  }

  public TiledTrainable(Tensor canvas, Layer filter, int tileSize, int padding) {
    this.canvas = canvas;
    this.filter = filter;
    Tensor filteredCanvas = this.filter.eval(canvas).getDataAndFree().getAndFree(0);
    assert 3 == filteredCanvas.getDimensions().length;
    int width = filteredCanvas.getDimensions()[0];
    int height = filteredCanvas.getDimensions()[1];
    int cols = (int) (Math.ceil((width - tileSize) * 1.0 / (tileSize - padding)) + 1);
    int rows = (int) (Math.ceil((height - tileSize) * 1.0 / (tileSize - padding)) + 1);
    if (cols != 1 || rows != 1) {
      int tileSizeX = (cols <= 1) ? width : (int) Math.ceil(((double) (width - padding) / cols) + padding);
      int tileSizeY = (rows <= 1) ? height : (int) Math.ceil(((double) (height - padding) / rows) + padding);
      logger.info(String.format(
          "Using Tile Size %s x %s to partition %s x %s png into %s x %s tiles",
          tileSizeX,
          tileSizeY,
          width,
          height,
          cols,
          rows
      ));
      selectors = new ImgTileSelectLayer[rows * cols];
      int index = 0;
      for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
          selectors[index++] = new ImgTileSelectLayer(
              tileSizeX,
              tileSizeY,
              col * (tileSizeX - padding),
              row * (tileSizeY - padding)
          );
        }
      }
      networks = IntStream.range(0, this.selectors.length)
          .mapToObj(i -> getNetwork(PipelineNetwork.build(1, filter, this.selectors[i]))).toArray(i -> new PipelineNetwork[i]);
    } else {
      selectors = null;
      networks = null;
    }
  }

  public void logTiles(final NotebookOutput log, Tensor image) {
    for (ImgTileSelectLayer selector : selectors) {
      log.p(String.format("Selector: %s", selector));
      log.eval(() -> {
        return selector.eval(image).getDataAndFree().getAndFree(0).toRgbImage();
      });
    }
  }

  @Override
  public PointSample measure(final TrainingMonitor monitor) {
    if (null == selectors) {
      Trainable trainable = new ArrayTrainable(getNetwork(filter), 1).setVerbose(isVerbose()).setMask(true).setData(Arrays.asList(new Tensor[][]{{canvas}}));
      PointSample measure = trainable.measure(monitor);
      trainable.freeRef();
      return measure;
    } else {
      Result canvasBuffer = filter.evalAndFree(new MutableResult(new UUID[]{canvasId}, canvas));
      final DeltaSet<UUID> delta = new DeltaSet<>();
      double result = IntStream.range(0, selectors.length).mapToDouble(i -> {
        Result tileInput = selectors[i].eval(canvasBuffer);
        Result tileOutput = networks[i].eval(tileInput);
        tileInput.freeRef();
        tileInput.getData().freeRef();
        tileOutput.accumulate(delta);
        Tensor tensor = tileOutput.getData().getAndFree(0);
        tileOutput.freeRef();
        double value = tensor.get(0);
        tensor.freeRef();
        return value;
      }).sum();
      canvasBuffer.getData().freeRef();
      canvasBuffer.freeRef();
      final StateSet<UUID> weights = new StateSet<>();
      weights.get(canvasId, canvas.getData()).set(canvas.getData()).freeRef();
      assert delta.getMap().keySet().stream().allMatch(x -> weights.getMap().containsKey(x));
      PointSample pointSample = new PointSample(delta, weights, result, 0, count.incrementAndGet());
      delta.freeRef();
      weights.freeRef();
      return pointSample;
    }
  }

  protected abstract PipelineNetwork getNetwork(Layer regionSelector);

  @Override
  public Layer getLayer() {
    return null;
  }

  public boolean isVerbose() {
    return verbose;
  }

  public TiledTrainable setVerbose(boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  @Override
  protected void _free() {
    if (null != selectors) Arrays.stream(selectors).forEach(ReferenceCountingBase::freeRef);
    filter.freeRef();
    super._free();
  }

  public NotebookOutput getLog() {
    return log;
  }

  public TiledTrainable setLog(NotebookOutput log) {
    this.log = log;
    return this;
  }
}
