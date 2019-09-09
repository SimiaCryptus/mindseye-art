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

import com.simiacryptus.mindseye.art.models.Inception5H;
import com.simiacryptus.mindseye.art.photo.FastPhotoStyleTransfer;
import com.simiacryptus.mindseye.art.photo.PixelGraph;
import com.simiacryptus.mindseye.art.photo.WCTUtil;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.SerialPrecision;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.WrapperLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.Util;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;

import javax.annotation.Nonnull;
import java.io.File;
import java.net.URI;
import java.util.Arrays;
import java.util.zip.ZipFile;

import static com.simiacryptus.mindseye.art.photo.FastPhotoStyleTransfer.transfer;

public class WCTTest extends NotebookReportBase {

//  private String contentImage = "file:///C:/Users/andre/Downloads/winter-with-snow-on-the-ground-landscape.jpg";
//  private String styleImage = "file:///C:/Users/andre/Downloads/wisconsin-77930_1280.jpg";

  private String contentImage = "file:///C:/Users/andre/Downloads/Herstmonceux_castle_summer_2005_(8414515391).jpg";
  private String styleImage = "file:///C:/Users/andre/Downloads/5212832572_4388ede3fc_o.jpg";

//  private String contentImage = "https://upload.wikimedia.org/wikipedia/commons/b/b0/Group_portrait_of_Civil_War_generals_n.d._%283200501542%29.jpg";
//  private String styleImage = "https://upload.wikimedia.org/wikipedia/commons/b/b6/Gilbert_Stuart_Williamstown_Portrait_of_George_Washington.jpg";
  private int imageSize = 800;
  private boolean verbose = true;

  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Components;
  }

  @Test
  public void test0() {
    run(this::test0);
  }

  private void test0(NotebookOutput log) {
    wct_test(
        log,
        new PipelineNetwork(1),
        new PipelineNetwork(1),
        contentImage(),
        styleImage()
    );
  }

  @NotNull
  private Tensor styleImage() {
    return Tensor.fromRGB(
        ImageUtil.resize(
            ImageUtil.getTensor(styleImage).toImage(),
            imageSize,
            true)
    );
  }

  @NotNull
  private Tensor contentImage() {
    return Tensor.fromRGB(
        ImageUtil.resize(
            ImageUtil.getTensor(contentImage).toImage(),
            imageSize,
            true)
    );
  }

  @Test
  public void test1() {
    run(this::test1);
  }

  private void test1(NotebookOutput log) {
    wct_test(
        log,
        VGG_WCT_Import.encode_1(),
        VGG_WCT_Import.decode_1(),
        contentImage(),
        styleImage()
    );
  }

  @Test
  public void test2() {
    run(this::test2);
  }

  private void test2(NotebookOutput log) {
    log.h1("Photo");
    wct_test(
        log,
        VGG_WCT_Import.encode_2(),
        VGG_WCT_Import.photo_decode_2(),
        contentImage(),
        styleImage()
    );
    log.h1("Centered");
    wct_test(
        log,
        VGG_WCT_Import.encode_2(),
        VGG_WCT_Import.decode_2(),
        contentImage(),
        styleImage()
    );
  }

  @Test
  public void test3() {
    run(this::test3);
  }

  private void test3(NotebookOutput log) {
    log.h1("Photo");
    wct_test(
        log,
        VGG_WCT_Import.encode_3(),
        VGG_WCT_Import.photo_decode_3(),
        contentImage(),
        styleImage()
    );
    log.h1("Centered");
    wct_test(
        log,
        VGG_WCT_Import.encode_3(),
        VGG_WCT_Import.decode_3(),
        contentImage(),
        styleImage()
    );
  }

  @Test
  public void test4() {
    run(this::test4);
  }

  private void test4(NotebookOutput log) {
    log.h1("Photo");
    wct_test(
        log,
        VGG_WCT_Import.encode_4(),
        VGG_WCT_Import.photo_decode_4(),
        contentImage(),
        styleImage()
    );
    log.h1("Centered");
    wct_test(
        log,
        VGG_WCT_Import.encode_4(),
        VGG_WCT_Import.decode_4(),
        contentImage(),
        styleImage()
    );
  }

  @Test
  public void test5() {
    run(this::test5);
  }

  private void test5(NotebookOutput log) {
    log.h1("Photo");
    wct_test(
        log,
        VGG_WCT_Import.encode_5(),
        VGG_WCT_Import.photo_decode_5(),
        contentImage(),
        styleImage()
    );
    log.h1("Centered");
    wct_test(
        log,
        VGG_WCT_Import.encode_5(),
        VGG_WCT_Import.decode_5(),
        contentImage(),
        styleImage()
    );
  }

  @Test
  public void wct_full() {
    run(this::wct_full);
  }

  private void wct_full(NotebookOutput log) {
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
    final Tensor content_4 = transfer(contentImage, styleImage, encode_4, decode_4);
    log.eval(() -> {
      return content_4.toImage();
    });

    final Layer encode_3 = VGG_WCT_Import.encode_3();
    final Layer decode_3 = VGG_WCT_Import.photo_decode_3();
    final Tensor content_3 = transfer(content_4, styleImage, encode_3, decode_3);
    log.eval(() -> {
      return content_3.toImage();
    });

    final Layer encode_2 = VGG_WCT_Import.encode_2();
    final Layer decode_2 = VGG_WCT_Import.photo_decode_2();
    final Tensor content_2 = transfer(content_3, styleImage, encode_2, decode_2);
    log.eval(() -> {
      return content_2.toImage();
    });

    final Layer encode_1 = VGG_WCT_Import.encode_1();
    final Layer decode_1 = VGG_WCT_Import.decode_1();
    final Tensor encodedContent = encode_1.eval(content_2).getDataAndFree().getAndFree(0);
    final Tensor encodedStyle = encode_1.eval(styleImage).getDataAndFree().getAndFree(0);
    final Tensor encodedTransformed = WCTUtil.applicator(encodedStyle).eval(encodedContent).getDataAndFree().getAndFree(0);
    final Tensor content_1 = decode_1.eval(encodedTransformed).getDataAndFree().getAndFree(0);
    log.eval(() -> {
      return content_1.toImage();
    });

    log.eval(() -> {
      return new PixelGraph(contentImage).smoothingTransform(1e-4, 1e-7).apply(content_1).toImage();
    });

  }

  @Test
  public void wct_api() {
    run(this::wct_api);
  }

  private void wct_api(NotebookOutput log) {
    Tensor contentImage = contentImage();
    Tensor styleImage = styleImage();
    log.eval(() -> {
      return contentImage.toImage();
    });
    log.eval(() -> {
      return styleImage.toImage();
    });

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
    if (verbose) {
      log.eval(() -> {
        return fastPhotoStyleTransfer.photoWCT_1(styleImage, contentImage).toImage();
      });
      log.eval(() -> {
        return fastPhotoStyleTransfer.photoWCT_2(styleImage, contentImage).toImage();
      });
      log.eval(() -> {
        return fastPhotoStyleTransfer.photoWCT_3(styleImage, contentImage).toImage();
      });
      log.eval(() -> {
        return fastPhotoStyleTransfer.photoWCT_4(styleImage, contentImage).toImage();
      });
      log.eval(() -> {
        return fastPhotoStyleTransfer.photoWCT(styleImage, contentImage).toImage();
      });
    }
    log.eval(() -> {
      return fastPhotoStyleTransfer.setLambda(1e-1).setEpsilon(1e-9)
          .apply(contentImage).apply(styleImage).toImage();
    });
  }

  private void wct_test(NotebookOutput log, Layer encoder, Layer decoder, Tensor contentImage, Tensor styleImage) {
    log.h2("Input");
    log.eval(() -> {
      return contentImage.toImage();
    });
    log.eval(() -> {
      return styleImage.toImage();
    });
    final Tensor originalFeatures = log.eval(() -> {
      return encoder.eval(contentImage).getDataAndFree().getAndFree(0);
    });

    log.h2("Encoding");
    log.eval(() -> {
      return Arrays.stream(originalFeatures.getDimensions()).mapToObj(Integer::toString).reduce((a, b) -> a + ", " + b).get();
    });
    final Tensor encodedStyle = log.eval(() -> {
      return encoder.eval(styleImage).getDataAndFree().getAndFree(0);
    });

    final Tensor restyledFeatures;
    if (verbose) {
      log.h2("Style Signal Stats");
      log.h3("Content");
      stats(log, encodedStyle);

      log.h3("Normalized");
      stats(log, log.eval(() -> {
        final Layer normalizer = WCTUtil.normalizer();
        final Tensor tensor = normalizer.eval(originalFeatures).getDataAndFree().getAndFree(0);
        normalizer.freeRef();
        return tensor;
      }));
      log.h3("Stylized");
      final PipelineNetwork styleNetwork = log.eval(() -> {
        return WCTUtil.applicator(encodedStyle);
      });
      restyledFeatures = log.eval(() -> {
        return styleNetwork.eval(originalFeatures).getDataAndFree().getAndFree(0);
      });
      stats(log, restyledFeatures);
    } else {
      restyledFeatures = WCTUtil.applicator(encodedStyle)
          .eval(originalFeatures).getDataAndFree().getAndFree(0);
    }

    log.h2("Result Images");
    System.setProperty("spark.master", "local[*]");
    System.setProperty("spark.app.name", getClass().getSimpleName());
    if (getNetwork(decoder).inputHandles.size() == 2) {
      final Tensor restyled = log.eval(() -> {
        return decoder.eval(restyledFeatures, contentImage).getDataAndFree().getAndFree(0);
      });
      log.eval(() -> {
        return restyled.toImage();
      });
      log.eval(() -> {
        return new PixelGraph(contentImage).smoothingTransform(1e-4, 1e-7).apply(restyled).toImage();
      });
      if (verbose) log.eval(() -> {
        return decoder.eval(originalFeatures, contentImage).getDataAndFree().getAndFree(0).toImage();
      });
    } else {
      final Tensor restyled = log.eval(() -> {
        return decoder.eval(restyledFeatures).getDataAndFree().getAndFree(0);
      });
      log.eval(() -> {
        return restyled.toImage();
      });
      log.eval(() -> {
        return new PixelGraph(contentImage).smoothingTransform(1e-4, 5e-2).apply(restyled).toImage();
      });
      if (verbose) log.eval(() -> {
        return decoder.eval(originalFeatures).getDataAndFree().getAndFree(0).toImage();
      });
    }
  }

  private PipelineNetwork getNetwork(Layer decoder) {
    if (decoder instanceof PipelineNetwork) return (PipelineNetwork) decoder;
    if (decoder instanceof WrapperLayer) return getNetwork(((WrapperLayer) decoder).getInner());
    throw new RuntimeException(decoder.getClass().getSimpleName());
  }

  private void stats(NotebookOutput log, Tensor normalFeatures) {
    final Tensor normalMeanSignal = log.eval(() -> {
      return WCTUtil.means(normalFeatures);
    });
    log.eval(() -> {
      return WCTUtil.rms(normalFeatures, normalMeanSignal);
    }).freeRef();
  }

  @Override
  protected Class<?> getTargetClass() {
    return Inception5H.class;
  }

}
