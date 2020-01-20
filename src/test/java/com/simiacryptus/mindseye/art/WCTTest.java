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
import com.simiacryptus.mindseye.art.photo.SmoothSolver_EJML;
import com.simiacryptus.mindseye.art.photo.WCTUtil;
import com.simiacryptus.mindseye.art.photo.affinity.GaussianAffinity;
import com.simiacryptus.mindseye.art.photo.affinity.MattingAffinity;
import com.simiacryptus.mindseye.art.photo.affinity.RasterAffinity;
import com.simiacryptus.mindseye.art.photo.affinity.RelativeAffinity;
import com.simiacryptus.mindseye.art.photo.cuda.RefOperator;
import com.simiacryptus.mindseye.art.photo.cuda.SmoothSolver_Cuda;
import com.simiacryptus.mindseye.art.photo.topology.ContentTopology;
import com.simiacryptus.mindseye.art.photo.topology.RadiusRasterTopology;
import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.art.photo.topology.SearchRadiusTopology;
import com.simiacryptus.mindseye.art.util.Plasma;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.SerialPrecision;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.WrapperLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.ref.wrappers.RefSystem;
import com.simiacryptus.util.Util;
import org.junit.Test;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.io.File;
import java.net.URI;
import java.util.concurrent.atomic.AtomicReference;
import java.util.zip.ZipFile;

import static com.simiacryptus.mindseye.art.photo.FastPhotoStyleTransfer.transfer;
import static com.simiacryptus.mindseye.art.photo.affinity.RasterAffinity.adjust;
import static com.simiacryptus.mindseye.art.photo.affinity.RasterAffinity.degree;
import static com.simiacryptus.mindseye.art.photo.topology.RadiusRasterTopology.getRadius;

public class WCTTest extends NotebookReportBase {

  //  private String contentImage = "file:///C:/Users/andre/Downloads/winter-with-snow-on-the-ground-landscape.jpg";
  //  private String styleImage = "file:///C:/Users/andre/Downloads/wisconsin-77930_1280.jpg";

  @Nonnull
  private String contentImage = "file:///C:/Users/andre/Downloads/pictures/E17-E.jpg";
  //"file:///C:/Users/andre/Downloads/Herstmonceux_castle_summer_2005_(8414515391).jpg";
  @Nonnull
  private String styleImage =
      //"file:///C:/Users/andre/Downloads/5212832572_4388ede3fc_o.jpg";
      "file:///C:/Users/andre/Downloads/pictures/1920x1080-kaufman_63748_5.jpg";

  //  private String contentImage = "https://upload.wikimedia.org/wikipedia/commons/b/b0/Group_portrait_of_Civil_War_generals_n.d._%283200501542%29.jpg";
  //  private String styleImage = "https://upload.wikimedia.org/wikipedia/commons/b/b6/Gilbert_Stuart_Williamstown_Portrait_of_George_Washington.jpg";
  private int imageSize = 600;
  private boolean verbose = false;

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
  public void test0() {
    run(this::test0);
  }

  @Test
  public void test1() {
    run(this::test1);
  }

  @Test
  public void test2() {
    run(this::test2);
  }

  @Test
  public void test3() {
    run(this::test3);
  }

  @Test
  public void test4() {
    run(this::test4);
  }

  @Test
  public void test5() {
    run(this::test5);
  }

  @Test
  public void wct_full() {
    run(this::wct_full);
  }

  @Test
  public void wct_api() {
    run(this::wct_api);
  }

  @Test
  public void photoBlur() {
    run(this::photoBlur);
  }

  @Test
  public void photoBlur_Survey() {
    run(this::photoBlur_Survey);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  WCTTest addRef() {
    return (WCTTest) super.addRef();
  }

  private void test0(@Nonnull NotebookOutput log) {
    wct_test(log, new PipelineNetwork(1), new PipelineNetwork(1), contentImage(), styleImage());
  }

  @Nonnull
  private Tensor styleImage() {
    return resize(ImageUtil.getTensor(styleImage));
  }

  @Nonnull
  private Tensor resize(@Nonnull Tensor tensor) {
    final Tensor resized = Tensor.fromRGB(ImageUtil.resize(tensor.toImage(), imageSize, true));
    tensor.freeRef();
    return resized;
  }

  @Nonnull
  private Tensor contentImage() {
    return resize(ImageUtil.getTensor(contentImage));
  }

  private void test1(@Nonnull NotebookOutput log) {
    wct_test(log, VGG_WCT_Import.encode_1(), VGG_WCT_Import.decode_1(), contentImage(), styleImage());
  }

  private void test2(@Nonnull NotebookOutput log) {
    log.h1("Photo");
    wct_test(log, VGG_WCT_Import.encode_2(), VGG_WCT_Import.photo_decode_2(), contentImage(), styleImage());
    log.h1("Centered");
    wct_test(log, VGG_WCT_Import.encode_2(), VGG_WCT_Import.decode_2(), contentImage(), styleImage());
  }

  private void test3(@Nonnull NotebookOutput log) {
    log.h1("Photo");
    wct_test(log, VGG_WCT_Import.encode_3(), VGG_WCT_Import.photo_decode_3(), contentImage(), styleImage());
    log.h1("Centered");
    wct_test(log, VGG_WCT_Import.encode_3(), VGG_WCT_Import.decode_3(), contentImage(), styleImage());
  }

  private void test4(@Nonnull NotebookOutput log) {
    log.h1("Photo");
    wct_test(log, VGG_WCT_Import.encode_4(), VGG_WCT_Import.photo_decode_4(), contentImage(), styleImage());
    log.h1("Centered");
    wct_test(log, VGG_WCT_Import.encode_4(), VGG_WCT_Import.decode_4(), contentImage(), styleImage());
  }

  private void test5(@Nonnull NotebookOutput log) {
    log.h1("Photo");
    wct_test(log, VGG_WCT_Import.encode_5(), VGG_WCT_Import.photo_decode_5(), contentImage(), styleImage());
    log.h1("Centered");
    wct_test(log, VGG_WCT_Import.encode_5(), VGG_WCT_Import.decode_5(), contentImage(), styleImage());
  }

  private void wct_full(@Nonnull NotebookOutput log) {
    Tensor contentImage = contentImage();
    Tensor styleImage = styleImage();

    log.eval(() -> {
      return contentImage.toImage();
    });
    log.eval(() -> {
      return styleImage.toImage();
    });

    final Layer encode_4 = VGG_WCT_Import.encode_4();
    final Layer decode_4 = VGG_WCT_Import.photo_decode_4();
    final Tensor content_4 = transfer(contentImage, styleImage, encode_4, decode_4, 1.0, 1.0);
    log.eval(() -> {
      return content_4.toImage();
    });

    final Layer encode_3 = VGG_WCT_Import.encode_3();
    final Layer decode_3 = VGG_WCT_Import.photo_decode_3();
    final Tensor content_3 = transfer(content_4, styleImage, encode_3, decode_3, 1.0, 1.0);
    log.eval(() -> {
      return content_3.toImage();
    });

    final Layer encode_2 = VGG_WCT_Import.encode_2();
    final Layer decode_2 = VGG_WCT_Import.photo_decode_2();
    final Tensor content_2 = transfer(content_3, styleImage, encode_2, decode_2, 1.0, 1.0);
    log.eval(() -> {
      return content_2.toImage();
    });

    final Layer encode_1 = VGG_WCT_Import.encode_1();
    final Layer decode_1 = VGG_WCT_Import.decode_1();
    final Tensor encodedContent = encode_1.eval(content_2).getData().get(0);
    final Tensor encodedStyle = encode_1.eval(styleImage).getData().get(0);
    final Tensor encodedTransformed = WCTUtil.applicator(encodedStyle, 1.0, 1.0).eval(encodedContent).getData().get(0);
    final Tensor content_1 = decode_1.eval(encodedTransformed).getData().get(0);
    log.eval(() -> {
      return content_1.toImage();
    });

    log.eval(() -> {
      final MattingAffinity affinity = new MattingAffinity(contentImage);
      return toImage(new SmoothSolver_EJML().solve(affinity.getTopology(), affinity, 1e-4).apply(content_1));
    });
  }

  private void wct_api(@Nonnull NotebookOutput log) {
    Tensor contentImage = contentImage();
    Tensor styleImage = styleImage();
    log.eval(() -> {
      return contentImage.toImage();
    });
    log.eval(() -> {
      return styleImage.toImage();
    });

    final FastPhotoStyleTransfer fastPhotoStyleTransfer = getFastPhotoStyleTransfer(log);
    if (verbose) {
      log.eval(() -> {
        return toImage(fastPhotoStyleTransfer.photoWCT_1(styleImage, contentImage));
      });
      log.eval(() -> {
        return toImage(fastPhotoStyleTransfer.photoWCT_2(styleImage, contentImage));
      });
      log.eval(() -> {
        return toImage(fastPhotoStyleTransfer.photoWCT_3(styleImage, contentImage));
      });
      log.eval(() -> {
        return toImage(fastPhotoStyleTransfer.photoWCT_4(styleImage, contentImage));
      });
      log.eval(() -> {
        return toImage(fastPhotoStyleTransfer.photoWCT(styleImage, contentImage));
      });
    }
    log.eval(() -> {
      final RefOperator<Tensor> operator = fastPhotoStyleTransfer.setLambda(1e-4).setEpsilon(1e-4).apply(contentImage);
      final BufferedImage image = toImage(operator.apply(styleImage));
      operator.freeRef();
      return image;
    });
    fastPhotoStyleTransfer.freeRef();
  }

  private FastPhotoStyleTransfer getFastPhotoStyleTransfer(@Nonnull NotebookOutput log) {
    File localFile = log.eval(() -> {
      return Util.cacheFile(new URI("https://simiacryptus.s3-us-west-2.amazonaws.com/photo_wct.zip"));
    });
    final FastPhotoStyleTransfer fastPhotoStyleTransfer;
    if (localFile == null || !localFile.exists()) {
      fastPhotoStyleTransfer = log.eval(() -> {
        return VGG_WCT_Import.newFastPhotoStyleTransfer();
      });
      final File out = new File(log.getResourceDir(), "photo_wct.zip");
      fastPhotoStyleTransfer.writeZip(out, SerialPrecision.Float);
      log.p(log.link(out, "Model Package"));
    } else {
      fastPhotoStyleTransfer = log.eval(() -> {
        return FastPhotoStyleTransfer.fromZip(new ZipFile(localFile));
      });
    }
    return fastPhotoStyleTransfer;
  }

  private void photoBlur(@Nonnull NotebookOutput log) {
    Tensor content = contentImage();
    log.eval(() -> {
      return content.toImage();
    });

    final int[] dimensions = content.getDimensions();
    final Tensor[] tensors = new Tensor[]{new Plasma().paint(dimensions[0], dimensions[1]),
        rawStyledContent(content, log), content};

    for (boolean selfRef : new boolean[]{true, false}) {
      for (boolean sqrt : new boolean[]{true, false}) {
        log.h1(RefString.format("SelfRef: %s, Sqrt: %s", selfRef, sqrt));

        log.h3("RadiusRasterTopology - MattingAffinity");
        test(log, log.eval(() -> {
          RasterTopology topology = new RadiusRasterTopology(dimensions, getRadius(1, 1), selfRef ? -1 : 0);
          RasterAffinity affinity = new MattingAffinity(content, topology).setGraphPower1(2).setMixing(0.5);
          if (sqrt)
            affinity = affinity
                .wrap((graphEdges, innerResult) -> adjust(graphEdges, innerResult, degree(innerResult), 0.5));
          return new SmoothSolver_Cuda().solve(topology, affinity, 1e-4);
        }), tensors);

        for (int contrast : new int[]{20, 50, 100}) {
          log.h2("Contrast: " + contrast);

          log.h3("SearchRadiusTopology");
          test(log, log.eval(() -> {
            ContentTopology topology = new SearchRadiusTopology(content).setSelfRef(selfRef).setVerbose(true);
            RasterAffinity affinity = new RelativeAffinity(content, topology).setContrast(contrast).setGraphPower1(2)
                .setMixing(0.5);
            if (sqrt)
              affinity = affinity
                  .wrap((graphEdges, innerResult) -> adjust(graphEdges, innerResult, degree(innerResult), 0.5));
            return new SmoothSolver_Cuda().solve(topology, affinity, 1e-4);
          }), tensors);

          log.h3("RadiusRasterTopology");
          test(log, log.eval(() -> {
            RasterTopology topology = new RadiusRasterTopology(dimensions, getRadius(1, 1), selfRef ? -1 : 0);
            RasterAffinity affinity = new RelativeAffinity(content, topology).setContrast(contrast).setGraphPower1(2)
                .setMixing(0.5);
            if (sqrt)
              affinity = affinity
                  .wrap((graphEdges, innerResult) -> adjust(graphEdges, innerResult, degree(innerResult), 0.5));
            return new SmoothSolver_Cuda().solve(topology, affinity, 1e-4);
          }), tensors);

          log.h3("RadiusRasterTopology - GaussianAffinity");
          test(log, log.eval(() -> {
            RasterTopology topology = new RadiusRasterTopology(dimensions, getRadius(1, 1), selfRef ? -1 : 0);
            RasterAffinity affinity = new GaussianAffinity(content, contrast, topology);
            if (sqrt)
              affinity = affinity
                  .wrap((graphEdges, innerResult) -> adjust(graphEdges, innerResult, degree(innerResult), 0.5));
            return new SmoothSolver_Cuda().solve(topology, affinity, 1e-4);
          }), tensors);
        }
      }
    }
  }

  private void photoBlur_Survey(@Nonnull NotebookOutput log) {
    Tensor content = contentImage();
    log.eval(() -> {
      return content.toImage();
    });

    final int[] dimensions = content.getDimensions();
    final Tensor[] tensors = new Tensor[]{new Plasma().paint(dimensions[0], dimensions[1]),
        rawStyledContent(content, log), content};

    for (boolean selfRef : new boolean[]{true, false}) {
      for (boolean sqrt : new boolean[]{true, false}) {
        log.h1(RefString.format("SelfRef: %s, Sqrt: %s", selfRef, sqrt));

        //        log.h3("RadiusRasterTopology - MattingAffinity");
        //        test(log, log.eval(() -> {
        //          RasterTopology topology = new RadiusRasterTopology(dimensions, getRadius(1, 1), selfRef ? -1 : 0);
        //          RasterAffinity affinity = new MattingAffinity(mask, topology).setGraphPower1(2).setMixing(0.5);
        //          if (sqrt) affinity = affinity.wrap((graphEdges, innerResult) -> adjust(graphEdges, innerResult, degree(innerResult), 0.5));
        //          return new SmoothSolver_Cuda().solve(topology, affinity, 1e-4);
        //        }), tensors);

        for (int contrast : new int[]{20, 50, 100}) {
          log.h2("Contrast: " + contrast);

          log.h3("SearchRadiusTopology");
          test(log, log.eval(() -> {
            ContentTopology topology = new SearchRadiusTopology(content).setSelfRef(selfRef).setVerbose(true);
            RasterAffinity affinity = new RelativeAffinity(content, topology).setContrast(contrast).setGraphPower1(2)
                .setMixing(0.5);
            if (sqrt)
              affinity = affinity
                  .wrap((graphEdges, innerResult) -> adjust(graphEdges, innerResult, degree(innerResult), 0.5));
            return new SmoothSolver_Cuda().solve(topology, affinity, 1e-4);
          }), tensors);

          //          log.h3("RadiusRasterTopology");
          //          test(log, log.eval(() -> {
          //            RasterTopology topology = new RadiusRasterTopology(dimensions, getRadius(1, 1), selfRef ? -1 : 0);
          //            RasterAffinity affinity = new RelativeAffinity(mask, topology).setContrast(contrast).setGraphPower1(2).setMixing(0.5);
          //            if (sqrt) affinity = affinity.wrap((graphEdges, innerResult) -> adjust(graphEdges, innerResult, degree(innerResult), 0.5));
          //            return new SmoothSolver_Cuda().solve(topology, affinity, 1e-4);
          //          }), tensors);
          //
          //          log.h3("RadiusRasterTopology - GaussianAffinity");
          //          test(log, log.eval(() -> {
          //            RasterTopology topology = new RadiusRasterTopology(dimensions, getRadius(1, 1), selfRef ? -1 : 0);
          //            RasterAffinity affinity = new GaussianAffinity(mask, contrast, topology);
          //            if (sqrt) affinity = affinity.wrap((graphEdges, innerResult) -> adjust(graphEdges, innerResult, degree(innerResult), 0.5));
          //            return new SmoothSolver_Cuda().solve(topology, affinity, 1e-4);
          //          }), tensors);
        }
      }
    }
  }

  private Tensor rawStyledContent(Tensor content, @Nonnull NotebookOutput log) {
    Tensor styleImage = styleImage();
    log.eval(() -> {
      return styleImage.toImage();
    });
    final FastPhotoStyleTransfer fastPhotoStyleTransfer = getFastPhotoStyleTransfer(log);
    final AtomicReference<Tensor> styledImage = new AtomicReference<>();
    log.eval(() -> {
      styledImage.set(fastPhotoStyleTransfer.setLambda(1e-4).setEpsilon(1e-4).photoWCT(styleImage, content));
      return styledImage.get().toImage();
    });
    Tensor styled = styledImage.get();
    return styled;
  }

  private void test(@Nonnull NotebookOutput log, @Nonnull RefOperator<Tensor> smoothingTransform, @Nonnull Tensor... styled) {
    try {
      RefArrays.stream(styled).filter(x -> x != null).forEach(tensor -> {
        log.eval(() -> {
          return smoothingTransform.apply(tensor).toImage();
        });
      });
      smoothingTransform.freeRef();
    } catch (Throwable e) {
      e.printStackTrace();
      RefSystem.gc();
    }
  }

  @Nonnull
  private BufferedImage toImage(@Nonnull Tensor tensor) {
    final BufferedImage bufferedImage = tensor.toImage();
    tensor.freeRef();
    return bufferedImage;
  }

  private void wct_test(@Nonnull NotebookOutput log, @Nonnull Layer encoder, @Nonnull Layer decoder, @Nonnull Tensor contentImage, @Nonnull Tensor styleImage) {
    log.h2("Input");
    log.eval(() -> {
      return contentImage.toImage();
    });
    log.eval(() -> {
      return styleImage.toImage();
    });
    final Tensor originalFeatures = log.eval(() -> {
      return encoder.eval(contentImage).getData().get(0);
    });

    log.h2("Encoding");
    log.eval(() -> {
      return RefUtil.get(RefArrays.stream(originalFeatures.getDimensions()).mapToObj(Integer::toString)
          .reduce((a, b) -> a + ", " + b));
    });
    final Tensor encodedStyle = log.eval(() -> {
      return encoder.eval(styleImage).getData().get(0);
    });

    final Tensor restyledFeatures;
    if (verbose) {
      log.h2("Style Signal Stats");
      log.h3("Content");
      stats(log, encodedStyle);

      log.h3("Normalized");
      stats(log, log.eval(() -> {
        final Layer normalizer = WCTUtil.normalizer();
        final Tensor tensor = normalizer.eval(originalFeatures).getData().get(0);
        normalizer.freeRef();
        return tensor;
      }));
      log.h3("Stylized");
      final PipelineNetwork styleNetwork = log.eval(() -> {
        return WCTUtil.applicator(encodedStyle);
      });
      restyledFeatures = log.eval(() -> {
        return styleNetwork.eval(originalFeatures).getData().get(0);
      });
      stats(log, restyledFeatures);
    } else {
      restyledFeatures = WCTUtil.applicator(encodedStyle).eval(originalFeatures).getData().get(0);
    }

    log.h2("Result Images");
    RefSystem.setProperty("spark.master", "local[*]");
    RefSystem.setProperty("spark.app.name", getClass().getSimpleName());
    if (getNetwork(decoder).inputHandles.size() == 2) {
      final Tensor restyled = log.eval(() -> {
        return decoder.eval(restyledFeatures, contentImage).getData().get(0);
      });
      log.eval(() -> {
        return restyled.toImage();
      });
      log.eval(() -> {
        final MattingAffinity affinity = new MattingAffinity(contentImage);
        return toImage(new SmoothSolver_EJML().solve(affinity.getTopology(), affinity, 1e-4).apply(restyled));
      });
      restyled.freeRef();
      if (verbose)
        log.eval(() -> {
          return toImage(decoder.eval(originalFeatures, contentImage).getData().get(0));
        });
    } else {
      final Tensor restyled = log.eval(() -> {
        return decoder.eval(restyledFeatures).getData().get(0);
      });
      log.eval(() -> {
        return restyled.toImage();
      });
      log.eval(() -> {
        final MattingAffinity affinity = new MattingAffinity(contentImage);
        return toImage(new SmoothSolver_EJML().solve(affinity.getTopology(), affinity, 1e-4).apply(restyled));
      });
      if (verbose)
        log.eval(() -> {
          return toImage(decoder.eval(originalFeatures).getData().get(0));
        });
    }
  }

  @Nonnull
  private PipelineNetwork getNetwork(Layer decoder) {
    if (decoder instanceof PipelineNetwork)
      return (PipelineNetwork) decoder;
    if (decoder instanceof WrapperLayer)
      return getNetwork(((WrapperLayer) decoder).getInner());
    throw new RuntimeException(decoder.getClass().getSimpleName());
  }

  private void stats(@Nonnull NotebookOutput log, Tensor normalFeatures) {
    final Tensor normalMeanSignal = log.eval(() -> {
      return WCTUtil.means(normalFeatures);
    });
    log.eval(() -> {
      return WCTUtil.rms(normalFeatures, normalMeanSignal);
    }).freeRef();
  }

}
