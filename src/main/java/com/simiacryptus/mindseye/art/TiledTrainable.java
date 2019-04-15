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

import com.google.common.util.concurrent.AtomicDouble;
import com.simiacryptus.lang.ref.ReferenceCounting;
import com.simiacryptus.lang.ref.ReferenceCountingBase;
import com.simiacryptus.mindseye.eval.BasicTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.ImgTileSelectLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
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
  private final Layer[] selectors;
  private final PipelineNetwork[] networks;
  @Nonnull
  private final Precision precision;
  AtomicInteger count = new AtomicInteger();
  private boolean verbose = false;
  private boolean mutableCanvas = true;

  public TiledTrainable(Tensor canvas, int tileSize, int padding) {
    this(canvas, tileSize, padding, Precision.Float);
  }

  public TiledTrainable(Tensor canvas, int tileSize, int padding, Precision precision) {
    this(canvas, new PipelineNetwork(1), tileSize, padding, precision);
  }

  public TiledTrainable(Tensor canvas, Layer filter, int tileSize, int padding) {
    this(canvas, filter, tileSize, padding, Precision.Float);
  }

  public TiledTrainable(Tensor canvas, Layer filter, int tileSize, int padding, @Nonnull Precision precision) {
    this(canvas, filter, tileSize, padding, precision, true);
  }

  public TiledTrainable(Tensor canvas, Layer filter, int tileSize, int padding, @Nonnull Precision precision, boolean largeTiles) {
    this.precision = precision;
    this.canvas = canvas;
    this.filter = filter.addRef();
    Tensor filteredCanvas = this.filter.eval(canvas).getDataAndFree().getAndFree(0);
    assert 3 == filteredCanvas.getDimensions().length;
    int width = filteredCanvas.getDimensions()[0];
    int height = filteredCanvas.getDimensions()[1];
    int cols = (int) (Math.ceil((width - tileSize) * 1.0 / (tileSize - padding)) + 1);
    int rows = (int) (Math.ceil((height - tileSize) * 1.0 / (tileSize - padding)) + 1);
    if (cols != 1 || rows != 1) {
      @NotNull ImgTileSelectLayer[] selectors = selectors(padding, width, height, tileSize, getPrecision());
      if (largeTiles) {
        this.selectors = Arrays.stream(selectors).map(ImgTileSelectLayer::getCompatibilityLayer).toArray(i -> new Layer[i]);
      } else {
        this.selectors = selectors;
      }
      networks = Arrays.stream(this.selectors)
          .map(selector -> PipelineNetwork.build(1, filter, selector))
          .map(this::getNetwork)
          .toArray(i -> new PipelineNetwork[i]);
    } else {
      selectors = null;
      networks = null;
    }
    logger.info("Trainable canvas ID: " + this.canvas.getId());
  }

  @NotNull
  public static ImgTileSelectLayer[] selectors(int padding, int width, int height, int tileSize, Precision precision) {
    int cols = (int) (Math.ceil((width - tileSize) * 1.0 / (tileSize - padding)) + 1);
    int rows = (int) (Math.ceil((height - tileSize) * 1.0 / (tileSize - padding)) + 1);
    int tileSizeX = (cols <= 1) ? width : (int) Math.ceil(((double) (width - padding) / cols) + padding);
    int tileSizeY = (rows <= 1) ? height : (int) Math.ceil(((double) (height - padding) / rows) + padding);
//    logger.info(String.format(
//        "Using Tile Size %s x %s to partition %s x %s png into %s x %s tiles",
//        tileSizeX,
//        tileSizeY,
//        width,
//        height,
//        cols,
//        rows
//    ));
    if (1 == cols && 1 == rows) {
      return new ImgTileSelectLayer[]{
          new ImgTileSelectLayer(
              width,
              height,
              0,
              0
          ).setPrecision(precision)
      };
    } else {
      ImgTileSelectLayer[] selectors = new ImgTileSelectLayer[rows * cols];
      int index = 0;
      for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
          selectors[index++] = new ImgTileSelectLayer(
              tileSizeX,
              tileSizeY,
              col * (tileSizeX - padding),
              row * (tileSizeY - padding)
          ).setPrecision(precision);
        }
      }
      return selectors;
    }
  }

  @Override
  public PointSample measure(final TrainingMonitor monitor) {
    assertAlive();
    if (null == selectors || 0 == selectors.length) {
      Trainable trainable = new BasicTrainable(PipelineNetwork.wrap(1,
          filter.addRef(),
          getNetwork(filter.addRef())
      ))
          .setMask(isMutableCanvas())
          .setData(Arrays.asList(new Tensor[][]{{canvas}}));
      PointSample measure = trainable.measure(monitor);
      trainable.freeRef();
      return measure;
    } else {
      Result canvasBuffer;
      if (isMutableCanvas()) {
        canvasBuffer = filter.evalAndFree(new MutableResult(canvas));
      } else {
        canvasBuffer = filter.evalAndFree(new ConstantResult(canvas));
      }
      AtomicDouble resultSum = new AtomicDouble(0);
      final DeltaSet<UUID> delta = IntStream.range(0, selectors.length).mapToObj(i -> {
        final DeltaSet<UUID> deltaSet = new DeltaSet<>();
        Result tileInput = selectors[i].eval(canvasBuffer);
        Result tileOutput = networks[i].eval(tileInput);
        tileInput.freeRef();
        tileInput.getData().freeRef();
        Tensor tensor = tileOutput.getData().getAndFree(0);
        assert 1 == tensor.length();
        resultSum.addAndGet(tensor.get(0));
        tileOutput.accumulate(deltaSet);
        tensor.freeRef();
        tileOutput.freeRef();
        return deltaSet;
      }).reduce((a, b) -> {
        a.addInPlace(b);
        b.freeRef();
        return a;
      }).get();
      canvasBuffer.getData().freeRef();
      canvasBuffer.freeRef();
      final StateSet<UUID> weights = new StateSet<>(delta);
      if (delta.getMap().containsKey(canvas.getId())) {
        weights.get(canvas.getId(), canvas.getData()).freeRef();
      }
      assert delta.getMap().keySet().stream().allMatch(x -> weights.getMap().containsKey(x));
      PointSample pointSample = new PointSample(delta, weights, resultSum.get(), 0, 1);
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
    if (null != selectors) Arrays.stream(selectors).forEach(ReferenceCounting::freeRef);
    if (null != networks) Arrays.stream(networks).forEach(ReferenceCounting::freeRef);

    filter.freeRef();
    super._free();
  }


  public boolean isMutableCanvas() {
    return mutableCanvas;
  }

  public TiledTrainable setMutableCanvas(boolean mutableCanvas) {
    this.mutableCanvas = mutableCanvas;
    return this;
  }

  public Precision getPrecision() {
    return precision;
  }

}