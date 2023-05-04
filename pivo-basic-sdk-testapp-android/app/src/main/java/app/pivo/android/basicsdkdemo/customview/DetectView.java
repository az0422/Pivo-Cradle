package app.pivo.android.basicsdkdemo.customview;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

import java.util.List;

import app.pivo.android.basicsdkdemo.tflite.Classifier;

public class DetectView extends View {
    private List<Classifier.Recognition> recognitions;
    private int rotate;

    public DetectView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }

    public void setRecognitions(List<Classifier.Recognition> recognitions, int rotate) {
        this.recognitions = recognitions;
        this.rotate = rotate;
        invalidate();
    }

    @Override
    public void onDraw(Canvas canvas) {
        if (recognitions == null) {
            canvas = new Canvas();
            super.onDraw(canvas);
            return;
        }

        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);

        for (Classifier.Recognition recognition : recognitions) {
                RectF rectF = recognition.getLocation();
                
                canvas.drawRect(
                        rectF.bottom,
                        rectF.right,
                        rectF.top,
                        rectF.left,
                        paint);
                canvas.drawText(
                        recognition.getId() + " " + recognition.getTitle(),
                        recognition.getLocation().left,
                        recognition.getLocation().top,
                        paint);
        }

        super.onDraw(canvas);
    }
}
