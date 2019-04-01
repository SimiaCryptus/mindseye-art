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
import com.simiacryptus.lang.ref.ReferenceCountingBase;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.PlaceholderLayer;
import com.simiacryptus.mindseye.layers.java.ImgTileAssemblyLayer;
import com.simiacryptus.mindseye.layers.java.ImgTileSelectLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.NullNotebookOutput;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Tiled trainable.
 */
public abstract class TiledTrainable extends ReferenceCountingBase implements Trainable {
  private final Tensor canvas;
  private int tileWidth = 600;
  private int tileHeight = 600;
  private NotebookOutput log = new NullNotebookOutput();
  private int padding = 0;
  private boolean verbose = false;
  private ImgTileSelectLayer[] selectors;
  private PipelineNetwork[] networks;

  public TiledTrainable(Tensor canvas) {
    this.canvas = canvas;
  }

  @Override
  public PointSample measure(final TrainingMonitor monitor) {
    assert 3 == canvas.getDimensions().length;
    int width = canvas.getDimensions()[0];
    int height = canvas.getDimensions()[1];
    int cols = (int) (Math.ceil((width - getTileWidth()) * 1.0 / (getTileWidth() - getPadding())) + 1);
    int rows = (int) (Math.ceil((height - getTileHeight()) * 1.0 / (getTileHeight() - getPadding())) + 1);
    if (cols == 1 && rows == 1) {
      networks = new PipelineNetwork[]{ getNetwork(new PipelineNetwork(1)) };
      return new ArrayTrainable(networks[0], 1).setVerbose(isVerbose()).setMask(true).setData(Arrays.asList(new Tensor[][]{{canvas}})).measure(monitor);
    } else {
      int tileSizeX = (cols <= 1) ? width : (int) Math.ceil(((double) (width - getPadding()) / cols) + getPadding());
      int tileSizeY = (rows <= 1) ? height : (int) Math.ceil(((double) (height - getPadding()) / rows) + getPadding());
      this.getLog().p(String.format(
          "Using Tile Size %s x %s to partition %s x %s png into %s x %s tiles",
          tileSizeX,
          tileSizeY,
          width,
          height,
          cols,
          rows
      ));

      if (null == selectors) {
        selectors = ImgTileSelectLayer.tileSelectors(getLog(), canvas, tileSizeX, tileSizeY, tileSizeX - getPadding(), tileSizeY - getPadding(), 0, 0);
      }
      if(null == networks) {
        networks = IntStream.range(0,selectors.length).mapToObj(i -> {
          return getNetwork(selectors[i]);
        }).toArray(i->new PipelineNetwork[i]);
      }

      AtomicDouble sum = new AtomicDouble();
      AtomicDouble rate = new AtomicDouble();
      AtomicInteger count = new AtomicInteger();
      List<Tensor> deltaList = IntStream.range(0,selectors.length).mapToObj(i -> {
        Tensor tile = selectors[i].eval(canvas).getDataAndFree().getAndFree(0);
        PointSample result = new ArrayTrainable(networks[i], 1).setVerbose(isVerbose()).setMask(true)
            .setData(Arrays.asList(new Tensor[][]{{tile}}))
            .measure(monitor);
        Delta<UUID> layerDelta = result.delta.stream().findAny().get();
        sum.addAndGet(result.sum);
        rate.addAndGet(result.rate);
        count.addAndGet(result.count);
        Tensor tensor = new Tensor(layerDelta.getDelta(), tile.getDimensions());
        result.freeRef();
        return tensor;
      }).collect(Collectors.toList());
      if (deltaList.size() != cols * rows) throw new AssertionError(deltaList.size() + " != " + cols + " * " + rows);

      PlaceholderLayer<double[]> placeholderLayer = new PlaceholderLayer<>(canvas.getData());
      ImgTileAssemblyLayer assemblyLayer = new ImgTileAssemblyLayer(cols, rows)
          .setPaddingX(getPadding()).setPaddingY(getPadding());
      Tensor assembled = assemblyLayer.eval(deltaList.toArray(new Tensor[]{})).getData().get(0);

      final DeltaSet<UUID> delta = new DeltaSet<>();
      if (canvas.getData().length != assembled.getData().length) throw new IllegalStateException(
          String.format(
              "%d != %d (%s != %s)",
              canvas.getData().length,
              assembled.getData().length,
              Arrays.toString(canvas.getDimensions()),
              Arrays.toString(assembled.getDimensions())
          ));
      delta.get(placeholderLayer.getId(), canvas.getData()).set(assembled.getData());
      final StateSet<UUID> weights = new StateSet<>();
      weights.get(placeholderLayer.getId(), canvas.getData()).set(canvas.getData());
      return new PointSample(delta, weights, sum.get(), rate.get(), count.get());
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
    super._free();
  }

  public int getTileWidth() {
    return tileWidth;
  }

  public TiledTrainable setTileWidth(int tileWidth) {
    this.tileWidth = tileWidth;
    return this;
  }

  public int getTileHeight() {
    return tileHeight;
  }

  public TiledTrainable setTileHeight(int tileHeight) {
    this.tileHeight = tileHeight;
    return this;
  }

  public int getPadding() {
    return padding;
  }

  public TiledTrainable setPadding(int padding) {
    this.padding = padding;
    return this;
  }

  public NotebookOutput getLog() {
    return log;
  }

  public TiledTrainable setLog(NotebookOutput log) {
    this.log = log;
    return this;
  }
}
