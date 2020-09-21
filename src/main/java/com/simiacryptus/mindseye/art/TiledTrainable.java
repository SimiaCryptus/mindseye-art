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
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;

/**
 * The type Tiled trainable.
 */
public abstract class TiledTrainable extends ReferenceCountingBase implements Trainable {

  private static final Logger logger = LoggerFactory.getLogger(TiledTrainable.class);

  /**
   * The Canvas.
   */
  public final Tensor canvas;
  /**
   * The Filter.
   */
  public final Layer filter;
  @Nullable
  private final Layer[] selectors;
  @Nullable
  private final PipelineNetwork[] networks;
  private final BasicTrainable basicTrainable;
  /**
   * The Precision.
   */
  @Nonnull
  public Precision precision;
  /**
   * The Mutable canvas.
   */
  public boolean mutableCanvas = true;

  /**
   * Instantiates a new Tiled trainable.
   *
   * @param canvas    the canvas
   * @param filter    the filter
   * @param tileSize  the tile size
   * @param padding   the padding
   * @param precision the precision
   */
  public TiledTrainable(Tensor canvas, @Nonnull Layer filter, int tileSize, int padding, @Nonnull Precision precision) {
    this(canvas, filter, tileSize, padding, precision, true);
  }

  /**
   * Instantiates a new Tiled trainable.
   *
   * @param canvas    the canvas
   * @param filter    the filter
   * @param tileSize  the tile size
   * @param padding   the padding
   * @param precision the precision
   * @param fade      the fade
   */
  public TiledTrainable(Tensor canvas, @Nonnull Layer filter, int tileSize, int padding, @Nonnull Precision precision,
                        boolean fade) {
    this.canvas = canvas.addRef();
    this.filter = filter;
    this.setPrecision(precision);
    Tensor filteredCanvas = Result.getData0(this.filter.eval(canvas.addRef()));
    assert 3 == filteredCanvas.getDimensions().length;
    int width = filteredCanvas.getDimensions()[0];
    int height = filteredCanvas.getDimensions()[1];
    filteredCanvas.freeRef();
    int cols = (int) (Math.ceil((width - tileSize) * 1.0 / (tileSize - padding)) + 1);
    int rows = (int) (Math.ceil((height - tileSize) * 1.0 / (tileSize - padding)) + 1);
    if (cols != 1 || rows != 1) {
      this.selectors = selectors(padding, width, height, tileSize, fade);
      networks = RefArrays.stream(RefUtil.addRef(selectors))
          .map(selector -> PipelineNetwork.build(1, this.filter.addRef(), selector))
          .map(regionSelector -> getNetwork(regionSelector))
          .toArray(i -> new PipelineNetwork[i]);
    } else {
      selectors = null;
      networks = null;
    }
    if (null == selectors || 0 == selectors.length) {
      this.basicTrainable = new BasicTrainable(PipelineNetwork.build(1,
          this.filter.addRef(),
          getNetwork(this.filter.addRef())));
      basicTrainable.setMask(isMutableCanvas());
      basicTrainable.setData(RefArrays.asList(new Tensor[][]{{canvas}}));
    } else {
      canvas.freeRef();
      this.basicTrainable = null;
    }
    logger.info("Trainable canvas ID: " + this.canvas.getId());
  }

  @Override
  public Layer getLayer() {
    return filter.addRef();
  }

  /**
   * Gets precision.
   *
   * @return the precision
   */
  @Nonnull
  public Precision getPrecision() {
    return precision;
  }

  /**
   * Sets precision.
   *
   * @param precision the precision
   */
  public void setPrecision(@Nonnull Precision precision) {
    this.precision = precision;
    MultiPrecision.setPrecision(filter.addRef(), precision);
  }

  /**
   * Is mutable canvas boolean.
   *
   * @return the boolean
   */
  public boolean isMutableCanvas() {
    return mutableCanvas;
  }

  /**
   * Sets mutable canvas.
   *
   * @param mutableCanvas the mutable canvas
   * @return the mutable canvas
   */
  @Nonnull
  public TiledTrainable setMutableCanvas(boolean mutableCanvas) {
    this.mutableCanvas = mutableCanvas;
    return this;
  }

  /**
   * Selectors layer [ ].
   *
   * @param padding  the padding
   * @param width    the width
   * @param height   the height
   * @param tileSize the tile size
   * @param fade     the fade
   * @return the layer [ ]
   */
  @Nonnull
  public static Layer[] selectors(int padding, int width, int height, int tileSize, boolean fade) {
    int cols = (int) (Math.ceil((width - tileSize) * 1.0 / (tileSize - padding)) + 1);
    int rows = (int) (Math.ceil((height - tileSize) * 1.0 / (tileSize - padding)) + 1);
    int tileSizeX = cols <= 1 ? width : (int) Math.ceil((double) (width - padding) / cols + padding);
    int tileSizeY = rows <= 1 ? height : (int) Math.ceil((double) (height - padding) / rows + padding);
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
            RefUtil.set(selectors, index++, tileSelectLayer);
          } else {
            int finalCol = col;
            int finalRow = row;
            Tensor coordSource = new Tensor(tileSizeX, tileSizeY, 1);
            Tensor mask = coordSource.mapCoords(c -> {
              int[] coords = c.getCoords();
              double v = 1.0;
              int left = coords[0];
              int right = tileSizeX - left;
              int top = coords[1];
              int bottom = tileSizeY - top;

              if (left < padding && finalCol > 0) {
                v *= (double) left / padding;
              } else {
                if (right < padding && finalCol < cols - 1) {
                  v *= (double) right / padding;
                }
              }
              if (top < padding && finalRow > 0) {
                v *= (double) top / padding;
              } else {
                if (bottom < padding && finalRow < rows - 1) {
                  v *= (double) bottom / padding;
                }
              }
              return v;
            });
            coordSource.freeRef();
            PipelineNetwork pipelineNetwork = new PipelineNetwork(1);
            InnerNode selectNode = pipelineNetwork.add(tileSelectLayer);
            pipelineNetwork.add(new ImgPixelGateLayer(), selectNode, pipelineNetwork.constValueWrap(mask)).freeRef();
            RefUtil.set(selectors, index++, pipelineNetwork);
          }
        }
      }
      return selectors;
    }
  }

  @Override
  public PointSample measure(final TrainingMonitor monitor) {
    assertAlive();
    if (null != basicTrainable) {
      return basicTrainable.measure(monitor);
    } else {
      final Result canvasBuffer;
      if (isMutableCanvas()) {
        canvasBuffer = this.filter.eval(new MutableResult(canvas.addRef()));
      } else {
        canvasBuffer = this.filter.eval(new ConstantResult(canvas.addRef()));
      }
      AtomicDouble resultSum = new AtomicDouble(0);
      final DeltaSet<UUID> delta = RefUtil.get(RefIntStream.range(0, selectors.length).mapToObj(i -> {
        final DeltaSet<UUID> deltaSet = new DeltaSet<>();
        Result tileInput = selectors[i].eval(canvasBuffer.addRef());
        assert networks != null;
        assert tileInput != null;
        Result tileOutput = networks[i].eval(tileInput);
        assert tileOutput != null;
        Tensor tensor = Result.getData0(tileOutput.addRef());
        assert 1 == tensor.length();
        resultSum.addAndGet(tensor.get(0));
        tileOutput.accumulate(deltaSet.addRef());
        tensor.freeRef();
        tileOutput.freeRef();
        return deltaSet;
      }).reduce((a, b) -> {
        a.addInPlace(b);
        return a;
      }));
      assert canvasBuffer != null;
      canvasBuffer.getData().freeRef();
      canvasBuffer.freeRef();
      final StateSet<UUID> weights = new StateSet<>(delta.addRef());
      RefMap<UUID, Delta<UUID>> deltaMap = delta.getMap();
      if (deltaMap.containsKey(canvas.getId())) {
        weights.get(canvas.getId(), canvas.addRef()).freeRef();
      }
      if (!weights.containsAll(deltaMap)) {
        delta.freeRef();
        weights.freeRef();
        throw new IllegalStateException();
      }
      return new PointSample(delta, weights, resultSum.get(), 0, 1);
    }
  }

  public void _free() {
    super._free();
    RefUtil.freeRef(basicTrainable);
    RefUtil.freeRef(selectors);
    RefUtil.freeRef(networks);
    filter.freeRef();
    canvas.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  TiledTrainable addRef() {
    return (TiledTrainable) super.addRef();
  }

  /**
   * Gets network.
   *
   * @param regionSelector the region selector
   * @return the network
   */
  protected abstract PipelineNetwork getNetwork(Layer regionSelector);

}
