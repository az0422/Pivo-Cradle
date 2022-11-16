/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package app.pivo.android.basicsdkdemo.tracking;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.text.TextUtils;
import android.util.Pair;
import android.util.TypedValue;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import app.pivo.android.basicsdkdemo.env.BorderedText;
import app.pivo.android.basicsdkdemo.env.ImageUtils;
import app.pivo.android.basicsdkdemo.env.Logger;
import app.pivo.android.basicsdkdemo.tflite.Classifier.Recognition;

/** A tracker that handles non-max suppression and matches existing objects to new detections. */
public class MultiBoxTracker {
  private static final float TEXT_SIZE_DIP = 18;
  private static final float MIN_SIZE = 16.0f;
  private static final int[] COLORS = {
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.CYAN,
    Color.MAGENTA,
    Color.WHITE,
    Color.parseColor("#55FF55"),
    Color.parseColor("#FFA500"),
    Color.parseColor("#FF8888"),
    Color.parseColor("#AAAAFF"),
    Color.parseColor("#FFFFAA"),
    Color.parseColor("#55AAAA"),
    Color.parseColor("#AA33AA"),
    Color.parseColor("#0D0068")
  };
  app.pivo.android.basicsdkdemo.tracking.Matrix sensorNoise = app.pivo.android.basicsdkdemo.tracking.Matrix.identity(2,2);
  app.pivo.android.basicsdkdemo.tracking.Matrix processNoise = app.pivo.android.basicsdkdemo.tracking.Matrix.identity(4,4);
  Function f = (Function) null;//by specifying null, you will use default function
  double minimumIOU = 0.65;  // SORT, 얼마나 IOU해야 같은 id로 볼것인지 지정.
  double newShapeWeight = 0.1;
  double threshold = 8; // SORT, 몇프레임까지 정보를 유지할지.
  double frame = 0;
  app.pivo.android.basicsdkdemo.tracking.Matrix initialVelocity = app.pivo.android.basicsdkdemo.tracking.Matrix.zero(2,1);
  app.pivo.android.basicsdkdemo.tracking.Matrix initialCovariance = app.pivo.android.basicsdkdemo.tracking.Matrix.identity(4,4);
  Sort s = new Sort(sensorNoise,processNoise,f,minimumIOU,newShapeWeight,threshold,initialVelocity,initialCovariance);
  final List<Pair<Float, RectF>> screenRects = new LinkedList<Pair<Float, RectF>>();
  private final Logger logger = new Logger();
  private final Queue<Integer> availableColors = new LinkedList<Integer>();
  private final List<TrackedRecognition> trackedObjects = new LinkedList<TrackedRecognition>();
  private final Paint boxPaint = new Paint();
  private final float textSizePx;
  private final BorderedText borderedText;
  private Matrix frameToCanvasMatrix;
  private int frameWidth;
  private int frameHeight;
  private int sensorOrientation;

  public MultiBoxTracker(final Context context) {
    for (final int color : COLORS) {
      availableColors.add(color);
    }

    boxPaint.setColor(Color.RED);
    boxPaint.setStyle(Style.STROKE);
    boxPaint.setStrokeWidth(10.0f);
    boxPaint.setStrokeCap(Cap.ROUND);
    boxPaint.setStrokeJoin(Join.ROUND);
    boxPaint.setStrokeMiter(100);

    textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
  }

  public synchronized void setFrameConfiguration(
      final int width, final int height, final int sensorOrientation) {
    frameWidth = width;
    frameHeight = height;
    this.sensorOrientation = sensorOrientation;
  }

  public synchronized void drawDebug(final Canvas canvas) {
    final Paint textPaint = new Paint();
    textPaint.setColor(Color.WHITE);
    textPaint.setTextSize(60.0f);

    final Paint boxPaint = new Paint();
    boxPaint.setColor(Color.RED);
    boxPaint.setAlpha(200);
    boxPaint.setStyle(Style.STROKE);

    for (final Pair<Float, RectF> detection : screenRects) {
      final RectF rect = detection.second;
      canvas.drawRect(rect, boxPaint);
      canvas.drawText("" + detection.first, rect.left, rect.top, textPaint);
      borderedText.drawText(canvas, rect.centerX(), rect.centerY(), "" + detection.first);
    }
  }

  public synchronized void trackResults(final List<Recognition> results, final long timestamp) {
    logger.i("Processing %d results from %d", results.size(), timestamp);
    processResults(results, timestamp);
  }

  private Matrix getFrameToCanvasMatrix() {
    return frameToCanvasMatrix;
  }

  public synchronized void draw(final Canvas canvas) {
    final boolean rotated = sensorOrientation % 180 == 90;
    final float multiplier =
        Math.min(
            canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
            canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
    frameToCanvasMatrix =
        ImageUtils.getTransformationMatrix(
            frameWidth,
            frameHeight,
            (int) (multiplier * (rotated ? frameHeight : frameWidth)),
            (int) (multiplier * (rotated ? frameWidth : frameHeight)),
            sensorOrientation,
            false);
    for (final TrackedRecognition recognition : trackedObjects) {
      final RectF trackedPos = new RectF(recognition.location);

      getFrameToCanvasMatrix().mapRect(trackedPos);
      boxPaint.setColor(recognition.color);

      float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;
      canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

      final String labelString =
          !TextUtils.isEmpty(recognition.title)
              ? String.format("%s %.2f", recognition.title, (100 * recognition.detectionConfidence))
              : String.format("%.2f", (100 * recognition.detectionConfidence));
      //            borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.top,
      // labelString);
      borderedText.drawText(
          canvas, trackedPos.left + cornerSize, trackedPos.top, labelString + "%", boxPaint);
    }
  }

  private void processResults(final List<Recognition> results, final long timestamp) {



    final List<Pair<Float, Recognition>> rectsToTrack = new LinkedList<Pair<Float, Recognition>>();

    screenRects.clear();
    final Matrix rgbFrameToScreen = new Matrix(getFrameToCanvasMatrix());

    for (final Recognition result : results) {
      if (result.getLocation() == null) {
        continue;
      }
      final RectF detectionFrameRect = new RectF(result.getLocation());

      final RectF detectionScreenRect = new RectF();
      rgbFrameToScreen.mapRect(detectionScreenRect, detectionFrameRect);

      logger.v(
          "Result! Frame: " + result.getLocation() + " mapped to screen:" + detectionScreenRect);

      screenRects.add(new Pair<Float, RectF>(result.getConfidence(), detectionScreenRect));

      if (detectionFrameRect.width() < MIN_SIZE || detectionFrameRect.height() < MIN_SIZE) {
        logger.w("Degenerate rectangle! " + detectionFrameRect);
        continue;
      }

      rectsToTrack.add(new Pair<Float, Recognition>(result.getConfidence(), result));
    }



    trackedObjects.clear();
    if (rectsToTrack.isEmpty()) {
      logger.v("Nothing to track, aborting.");
      return;
    }

    ArrayList<Measurement> arr_dum = new ArrayList<Measurement>();
    for (final Pair<Float, Recognition> potential : rectsToTrack){
      final Recognition recog = potential.second;
      float x = recog.getLocation().left;
      float y = recog.getLocation().top;
      float width = recog.getLocation().width();
      float height = recog.getLocation().height();
      Measurement m = new Measurement(x,y,width,height,recog.getTitle(),0);
      arr_dum.add(m);
    }




    Measurement[] tmp = new Measurement[arr_dum.size()];
    for(int i=0; i<arr_dum.size(); i++){
      tmp[i] = arr_dum.get(i);}
    s.predictPhase(frame);
    s.updatePhase(tmp,frame);
    TrackedObject[] track_result = s.getTracked();
    frame = frame + 1;
    boolean dummy = false;

    for (final TrackedObject potential : track_result) {
      final TrackedRecognition trackedRecognition = new TrackedRecognition();
      for(final Pair<Float, Recognition> potential2 : rectsToTrack){
        trackedRecognition.location = new RectF(potential2.second.getLocation());
        trackedRecognition.detectionConfidence = potential2.second.getConfidence();
        rectsToTrack.remove(0);
        if (dummy == false){
          break;
        }
      }


      trackedRecognition.title = potential.objectType + " " + potential.name;
      trackedRecognition.color = COLORS[trackedObjects.size()];
      trackedObjects.add(trackedRecognition);




      if (trackedObjects.size() >= COLORS.length) {
        break;
      }
    }







  }

  private static class TrackedRecognition {
    RectF location;
    float detectionConfidence;
    int color;
    String title;
  }
}
