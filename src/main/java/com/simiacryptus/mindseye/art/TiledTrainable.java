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
import com.simiacryptus.mindseye.eval.BasicTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.java.ImgPixelGateLayer;
import com.simiacryptus.mindseye.layers.java.ImgTileSelectLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.UUID;

public abstract class TiledTrainable extends ReferenceCountingBase implements Trainable {

  private static final Logger logger = LoggerFactory.getLogger(TiledTrainable.class);

  public final Tensor canvas;
  public final Layer filter;
  @Nullable
  private final Layer[] selectors;
  @Nullable
  private final PipelineNetwork[] networks;
  private final Singleton<PipelineNetwork> networkSingleton = new Singleton<>();
  @Nonnull
  public Precision precision;
  public boolean mutableCanvas = true;

  public TiledTrainable(Tensor canvas, int tileSize, int padding) {
    this(canvas, tileSize, padding, Precision.Float);
  }

  public TiledTrainable(Tensor canvas, int tileSize, int padding, @Nonnull Precision precision) {
    this(canvas, new PipelineNetwork(1), tileSize, padding, precision);
  }

  public TiledTrainable(Tensor canvas, @Nonnull Layer filter, int tileSize, int padding) {
    this(canvas, filter, tileSize, padding, Precision.Float);
  }

  public TiledTrainable(Tensor canvas, @Nonnull Layer filter, int tileSize, int padding, @Nonnull Precision precision) {
    this(canvas, filter, tileSize, padding, precision, true);
  }

  public TiledTrainable(Tensor canvas, @Nonnull Layer filter, int tileSize, int padding, @Nonnull Precision precision,
                        boolean fade) {
    this.setPrecision(precision);
    this.canvas = canvas;
    this.filter = filter.addRef();
    Tensor filteredCanvas = this.filter.eval(canvas).getData().get(0);
    assert 3 == filteredCanvas.getDimensions().length;
    int width = filteredCanvas.getDimensions()[0];
    int height = filteredCanvas.getDimensions()[1];
    int cols = (int) (Math.ceil((width - tileSize) * 1.0 / (tileSize - padding)) + 1);
    int rows = (int) (Math.ceil((height - tileSize) * 1.0 / (tileSize - padding)) + 1);
    if (cols != 1 || rows != 1) {
      this.selectors = selectors(padding, width, height, tileSize, fade);
      networks = RefArrays.stream(this.getSelectors())
          .map(selector -> PipelineNetwork.build(1, filter.addRef(), selector)).map(this::getNetwork)
          .toArray(i -> new PipelineNetwork[i]);
    } else {
      selectors = null;
      networks = null;
    }
    logger.info("Trainable canvas ID: " + this.canvas.getId());
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    return null;
  }

  @Nonnull
  public Precision getPrecision() {
    return precision;
  }

  public void setPrecision(@Nonnull Precision precision) {
    this.precision = precision;
    MultiPrecision.setPrecision((DAGNetwork) filter, precision);
  }

  @Nullable
  public Layer[] getSelectors() {
    return selectors;
  }

  public boolean isMutableCanvas() {
    return mutableCanvas;
  }

  @Nonnull
  public TiledTrainable setMutableCanvas(boolean mutableCanvas) {
    this.mutableCanvas = mutableCanvas;
    return this;
  }

  @Nonnull
  public static Layer[] selectors(int padding, int width, int height, int tileSize, boolean fade) {
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
      return new ImgTileSelectLayer[]{new ImgTileSelectLayer(width, height, 0, 0)};
    } else {
      Layer[] selectors = new Layer[rows * cols];
      int index = 0;
      for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
          ImgTileSelectLayer tileSelectLayer = new ImgTileSelectLayer(tileSizeX, tileSizeY, col * (tileSizeX - padding),
              row * (tileSizeY - padding));
          if (!fade) {
            selectors[index++] = tileSelectLayer;
          } else {
            int finalCol = col;
            int finalRow = row;
            Tensor mask = new Tensor(tileSizeX, tileSizeY, 1).mapCoords(c -> {
              int[] coords = c.getCoords();
              double v = 1.0;
              if (coords[0] < padding && finalCol > 0) {
                v *= coords[0] / padding;
              } else if ((tileSizeX - coords[0]) < padding && finalCol < (cols - 1)) {
                v *= (double) (tileSizeX - coords[0]) / padding;
              }
              if (coords[1] < padding && finalRow > 0) {
                v *= (double) coords[1] / padding;
              } else if ((tileSizeY - coords[1]) < padding && finalRow < (rows - 1)) {
                v *= (double) (tileSizeY - coords[1]) / padding;
              }
              return v;
            });
            PipelineNetwork pipelineNetwork = new PipelineNetwork(1);
            InnerNode selectNode = pipelineNetwork.add(tileSelectLayer);
            pipelineNetwork.add(new ImgPixelGateLayer(), selectNode, pipelineNetwork.constValueWrap(mask)).freeRef();
            selectors[index++] = pipelineNetwork;
          }
        }
      }
      return selectors;
    }
  }

  @Nullable
  public static @SuppressWarnings("unused")
  TiledTrainable[] addRefs(@Nullable TiledTrainable[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TiledTrainable::addRef)
        .toArray((x) -> new TiledTrainable[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  TiledTrainable[][] addRefs(@Nullable TiledTrainable[][] array) {
    return RefUtil.addRefs(array);
  }

  @Override
  public PointSample measure(final TrainingMonitor monitor) {
    assertAlive();
    final Layer filter = this.filter.addRef();
    if (null == getSelectors() || 0 == getSelectors().length) {
      BasicTrainable trainable = new BasicTrainable(PipelineNetwork.build(1, filter.addRef(),
          networkSingleton.getOrInit(() -> getNetwork(filter.addRef())).addRef()));
      trainable.setMask(isMutableCanvas());
      trainable.setData(RefArrays.asList(new Tensor[][]{{canvas}}));
      PointSample measure = trainable.measure(monitor);
      trainable.freeRef();
      filter.freeRef();
      return measure;
    } else {
      Result canvasBuffer;
      if (isMutableCanvas()) {
        canvasBuffer = filter.eval(new MutableResult(canvas));
      } else {
        canvasBuffer = filter.eval(new ConstantResult(canvas));
      }
      filter.freeRef();
      AtomicDouble resultSum = new AtomicDouble(0);
      final DeltaSet<UUID> delta = RefUtil.get(RefIntStream.range(0, getSelectors().length).mapToObj(i -> {
        final DeltaSet<UUID> deltaSet = new DeltaSet<>();
        Result tileInput = getSelectors()[i].eval(canvasBuffer);
        assert networks != null;
        Result tileOutput = networks[i].eval(tileInput);
        assert tileInput != null;
        tileInput.freeRef();
        tileInput.getData().freeRef();
        assert tileOutput != null;
        Tensor tensor = tileOutput.getData().get(0);
        assert 1 == tensor.length();
        resultSum.addAndGet(tensor.get(0));
        tileOutput.accumulate(deltaSet);
        tensor.freeRef();
        tileOutput.freeRef();
        return deltaSet;
      }).reduce((a, b) -> {
        a.addInPlace(b);
        this.addRef();
        b.freeRef();
        return a;
      }));
      assert canvasBuffer != null;
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

  public void _free() {
    if (null != getSelectors())
      RefArrays.stream(getSelectors()).forEach(ReferenceCounting::freeRef);
    if (null != networks)
      RefArrays.stream(networks).forEach(ReferenceCounting::freeRef);
    networkSingleton.freeRef();
    filter.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  TiledTrainable addRef() {
    return (TiledTrainable) super.addRef();
  }

  protected abstract PipelineNetwork getNetwork(Layer regionSelector);

}
