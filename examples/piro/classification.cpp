#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <math.h>
#include <vector>
#include <numeric>
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using std::cout;
using std::endl;
using cv::Mat;
using std::max;

const string LOG_DIR = "/Users/kareth/code/studia/piro/tmp/";
const bool LOG_IMAGES = true;

void LogImg(const Mat& img, const string& filename) {
  if (LOG_IMAGES)
    cv::imwrite(LOG_DIR + filename + ".jpg", img);
}

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

enum SymbolType {
  N1, N2, N4, N8,  // Notes
  R1, R2, R4, R8,  // Rests
  TC, BC,          // Clefs
  BAR,             // Bars
  FA, SA           // Accidentals
};

struct Position {
  Position(int x) : index_(x) {}
  // IMPLICIT
  operator int() const { return index_; }

  string ToKeyAndOctave() const {
    int octave = 4;
    const vector<string> keys = {"c", "d", "e", "f", "g", "a", "b"};
    int pos = index_ + 2; // 0 = e/4;
    while (pos >= keys.size() && octave <= 6) { pos -= keys.size(); octave++; }
    while (pos < 0            && octave >= 3) { pos += keys.size(); octave--; }
    return keys[pos] + "/" + std::to_string(octave);
  }
  int index_ = 0;
};

struct Symbol {
  Symbol(SymbolType type, int position = 0) : type(type), position(position) {}
  SymbolType type;
  Position position;
};

std::map<string, SymbolType> label_to_symbol_type = {
  {"n1 Whole note", N1},
  {"n2 Half note", N2},
  {"n4 Quarter note", N4},
  {"n8 Eighth note", N8},
  {"r1 Whole rest", R1},
  {"r2 Half rest", R2},
  {"r4 Quarter rest", R4},
  {"r8 Eighth rest", R8},
  {"tc Treble clef", TC},
  {"bc Bass clef", BC},
  {"bar Bar line", BAR},
  {"fa Flat accidental", FA},
  {"sa Sharp accidental", SA}
};

class Line {
 public:
  Line(const vector<Symbol> symbols) : symbols_(symbols) {}
  Line(const vector<vector<Symbol>> symbol_groups) {
    int group_id = 0;
    for (const auto& g : symbol_groups) {
      if (g.size() == 1) {
        symbols_.push_back(g[0]);
        groups_.push_back(-1);
      } else {
        for (const auto& s : g) {
          symbols_.push_back(s);
          groups_.push_back(group_id);
        }
        group_id++;
      }
    }
  }
  const vector<Symbol>& symbols() const { return symbols_; }
  const vector<int>& groups() const { return groups_; }
  void Print() const { for (auto s : symbols_) cout << int(s.type) << ", "; cout << endl; }
 private:
  vector<Symbol> symbols_;
  vector<int> groups_;
};

// Helper method
vector<int> SumPixelsOverRows(const Mat& m) {
  vector<int> row_pixels(m.rows);
  for (int x = 0; x < m.cols; x++) {
    for (int y = 0; y < m.rows; y++) {
      uchar pixel = m.at<uchar>(y, x);
      row_pixels[y] += pixel;
    }
  }
  return row_pixels;
}

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

class LineReader {
 public:
  LineReader(Classifier* classifier, const string& filename)
      : classifier_(classifier) {
    src_ = cv::imread(filename, -1);
    if (!src_.data) std::cerr << "Problem loading image!!!" << endl;
    if (src_.channels() == 3) {
      cv::cvtColor(src_, gray_, CV_BGR2GRAY);
    } else {
      gray_ = src_.clone();
    }
  }

  Line Read() {
    SeparateLines();
    SeparateSymbols();

    vector<vector<Symbol>> symbols;
    vector<pair<int, int>> ranges;
    auto process_range = [&](int x, int y) {
      if (!IsTrash(notes_, x, y))
        ranges.emplace_back(x, y);
    };

    ExtractRanges(notes_, process_range);

    using NoteBox = vector<pair<int, int>>;

    vector<NoteBox> note_boxes;

    for (int i = 0; i < ranges.size() - 1; i++) {
      if (ranges[i+1].first - ranges[i].second < line_distance_ / 2) {
        ranges[i].second = ranges[i+1].second;
        ranges.erase(ranges.begin() + i + 1);
        i--;
        continue;
      }
      note_boxes.push_back(SplitToNoteBoxes(notes_, ranges[i]));
    }

    Mat rgb;
    cvtColor(notes_, rgb, CV_GRAY2RGB);
    bool color = 0;
    for (const auto& box : note_boxes) {
      bool multiple_notes = (box.size() > 1);

      vector<Symbol> symbol_group;

      for (const auto& range : box) {
        int x = range.first, y = range.second;
        Mat note = CropNote(notes_, x, y);

        std::vector<Prediction> predictions;

        if (multiple_notes) {
          Symbol sym(N8);
          DetectHeight(&sym, note);
          symbol_group.emplace_back(sym);
        } else {
          auto pred = classifier_->Classify(note, 5);
          Symbol sym(label_to_symbol_type[pred[0].first]);
          DetectHeight(&sym, note);
          std::cout << cropped_notes_-1 << " " << pred[0].first << " " << pred[0].second << "\% pos: " << int(sym.position) << std::endl;
          symbol_group.emplace_back(sym);
        }

        for (int c = x; c <= y ; c++) {
          for (int r = 0; r < rgb.rows; r++) {
            rgb.at<cv::Vec3b>(r, c)[color] = 0;
            if (multiple_notes)
              rgb.at<cv::Vec3b>(r, c)[2] = 0;
          }
        }
        color ^= 1;
      }
      symbols.push_back(symbol_group);
    }
    LogImg(rgb, "argb");

    return Line(symbols);
  }

 private:

  vector<pair<int, int>> SplitToNoteBoxes(const Mat& m, const pair<int, int>& range) {
    //return {range};
    vector<int> head_areas;

    for (int x = range.first; x <= range.second; x++) {
      int longest = 0, l = 0;
      for (int y = 0; y < m.rows; y++) {
        if (m.at<uchar>(y, x) > 50) { // wgute
          longest = max(longest, y - l);
          l = y;
        }
      }
      longest = max(longest, m.rows - 1 - l);
      //printf("%d ",longest);

      if (longest > line_distance_ / 2 &&
          longest * 2 <= line_distance_ * 3)
        head_areas.push_back(x);
    }

    vector<pair<int, int>> heads;

    for (int i = 0; i < head_areas.size(); i++) {
      int l = i, r = i;
      while (r + 1 < head_areas.size() && head_areas[r+1] - head_areas[r] < 3) r++;
      if (r - l + 1 > line_distance_ / 2) {
        heads.emplace_back(head_areas[l], head_areas[r]);
      }
      i = r + 1;
    }

    printf("\nRange (%d, %d) -> ",range.first, range.second);
    //for (int a : head_areas) printf("%d ",a); printf("\n");

    for (const auto& h : heads) printf("<%d %d> ", h.first, h.second); printf("\n");

    if (heads.size() > 1) {
      return heads;
    } else {
      return {range};
    }
  }

  bool IsTrash(const Mat& m, int a, int b) {
    int sum = 0;
    for (int x = a; x <= b; x++)
      for (int y = 0; y < m.rows; y++)
        if (m.at<uchar>(y, x) < 50) sum++;
    int all = (b - a + 1) * m.rows;
    printf("%d of %d black\n", sum, all);
    return double(sum) / double(all) < 0.03;
  }

  void DetectHeight(Symbol* symbol, const Mat& note_mat) {
    auto t = symbol->type;
    // Notes
    if (t == N1 || t == N2 || t == N4 || t == N8) {
      auto row_pixels = SumPixelsOverRows(note_mat);
      // Count black pixels, not white
      for (int& p : row_pixels) p = 255 * note_mat.cols - p;
      // threshold on 30% of max value
      int threshold = *std::max_element(row_pixels.begin(), row_pixels.end()) * 0.3;

      for (int& p : row_pixels) if (p < threshold) p = 0;

      // Find largest segment of pixels
      int best = 0, bestsize = 0, now = 0;
      for (int i = 0; i < row_pixels.size(); i++) {
        if (row_pixels[i] == 0) {
          if (i - now > bestsize) {
            bestsize = i - now;
            best = now;
          }
          now = i+1;
        }
      }
      if (row_pixels.size() - now > bestsize) {
        bestsize = row_pixels.size() - now;
        best = now;
      }

      // Remove lines other than largest segment
      for (int i = 0; i < row_pixels.size(); i++)
        if (i < best || i >= best + bestsize)
          row_pixels[i] = 0;

      vector<int> lines(note_mat.rows);
      std::iota(lines.begin(), lines.end(), 0);
      double weighted_line = std::inner_product(row_pixels.begin(), row_pixels.end(), lines.begin(), 0);
      weighted_line /= std::accumulate(row_pixels.begin(), row_pixels.end(), 0);

      symbol->position = GetPosition(weighted_line);

      // LOG
      Mat cp = note_mat.clone();
      for (int x = 0; x < cp.cols; x++) cp.at<uchar>(int(weighted_line), x) = uchar(100);
      LogImg(cp, "crop_" + std::to_string(cropped_notes_ - 1) + "+weight");
    }
  }

  Mat CropNote(const Mat& m, int start, int end) {
    Mat cp = m.clone();
    Mat crop = cp(cv::Rect(start, 0, end-start, notes_.rows));
    // TODO(aszczepanski): adjust border size
    copyMakeBorder(crop.clone(), crop, 0, 0, 20, 20, cv::BORDER_CONSTANT, cv::Scalar(255));
    LogImg(crop, "crop_" + std::to_string(cropped_notes_++));
    return crop;
  }

  void ExtractRanges(const Mat& m, std::function<void(int, int)> emit) {
    int l = 0, r = 0;
    for (int x = 0; x < m.cols; x++) {
      bool has_pixel = false;
      for (int y = 0; y < m.rows; y++) {
        uchar pixel = m.at<uchar>(y, x);
        // TODO(pzurkowski) do normal thresholding and work on 0/1 image. Dilitate too?
        if (pixel < 50) has_pixel = true;
      }
      if (has_pixel) {
        r++;
      } else {
        if (l < r) emit(l, r);
        r++;
        l = r;
      }
    }

    if (l < r) emit(l, r);
  }

  void SeparateLines() {
    LogImg(gray_, "gray");

    Mat bw;
    cv::adaptiveThreshold(~gray_, bw, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 25, -2);
    LogImg(bw, "bit");

    horizontal_ = bw.clone();
    vertical_ = bw.clone();
    // Specify size on horizontal axis
    int horizontalsize = horizontal_.cols / 30;
    // Create structure element for extracting horizontal lines through morphology operations
    Mat horizontalStructure = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(horizontalsize,1));
    // Apply morphology operations
    cv::erode(horizontal_, horizontal_, horizontalStructure, cv::Point(-1, -1));
    cv::dilate(horizontal_, horizontal_, horizontalStructure, cv::Point(-1, -1));
    LogImg(horizontal_, "horizontal");

    ExtractLinePositions(horizontal_);

    // Specify size on vertical axis
    int verticalsize = vertical_.rows / 30;
    // Create structure element for extracting vertical lines through morphology operations
    Mat verticalStructure = cv::getStructuringElement(cv::MORPH_RECT, cv::Size( 1,verticalsize));
    // Apply morphology operations
    cv::erode(vertical_, vertical_, verticalStructure, cv::Point(-1, -1));
    cv::dilate(vertical_, vertical_, verticalStructure, cv::Point(-1, -1));
    // Show extracted vertical lines
    cv::bitwise_not(vertical_, vertical_);
    LogImg(vertical_, "vertical");
  }

  void ExtractLinePositions(const Mat& m) {
    auto row_pixels = SumPixelsOverRows(m);

    // threshold on 30% of max value
    int threshold = *std::max_element(row_pixels.begin(), row_pixels.end()) * 0.3;

    vector<pair<int, int>> candidates;
    for (int r = 0; r < row_pixels.size(); r++)
      if (row_pixels[r] > threshold)
        candidates.emplace_back(r, row_pixels[r]);

    //for (const auto& c : candidates)
    //  printf ("%d %d\n", c.first, c.second);

    if (candidates.size() > 5) {
      for (int i = 0; i < candidates.size() - 1; i++) {
        int l = i, r = i;
        int sec = candidates[l].second;
        while (r < candidates.size() - 1 &&
               candidates[r+1].first - candidates[r].first <= 1) {
          r++;
          sec = std::min(sec, candidates[r+1].second);
        }
        //printf("sum %d %d\n", candidates[l].first, candidates[r].first);
        candidates[l].second = sec;
        candidates[l].first = (candidates[l].first + candidates[r].first) / 2;

        candidates.erase(candidates.begin() + l + 1, candidates.begin() + r + 1);
      }
    }

    while (candidates.size() > 5) {
      int m = 0;
      for (int i = 0; i < candidates.size(); i++) {
        if (candidates[i].second > candidates[m].second)
          m = i;
      }
      candidates.erase(candidates.begin() + m);
    }

    //for (const auto& c : candidates)
    //  printf ("%d %d\n", c.first, c.second);
    for (const auto& c : candidates) line_positions_.push_back(c.first);
    line_distance_ = double(line_positions_.back() - line_positions_[0])/ 4.;

    // Set thresholds
    for (int pos : line_positions_) {
      line_thresholds_.push_back(double(pos) - line_distance_ / 4.);
      line_thresholds_.push_back(double(pos) + line_distance_ / 4.);
    }

    for (double t : line_thresholds_) printf("%lf, ",t);
    printf("\n");

    printf("Lines on pixels: ");
    for (const auto& c : candidates) printf("%d, ", c.first);
    printf("\n");
  }

  void SeparateSymbols() {
    /* Mat lines;
    vertical_.copyTo(lines, horizontal_);
    notes_ = gray_ + lines + lines;
    LogImg(notes_, "notes"); */

    Mat lines;
    vertical_.copyTo(lines, horizontal_);
    Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));
    cv::dilate(lines, lines, kernel, cv::Point(-1, -1));
    LogImg(lines, "lines");
    notes_ = gray_ + lines + lines;

    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(notes_, notes_, kernel, cv::Point(-1, -1));

    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(notes_, notes_, kernel, cv::Point(-1, -1));

    notes_ = notes_ > 128;
    LogImg(notes_, "notes");

    //LogImg(notes_bw, "notes_bw");
  }

  Position GetPosition(double position) {
    if (position < line_thresholds_[0]) {
      position = line_thresholds_[0] - position;
      double step = line_distance_ / 2;
      return 8 + ceil(position / step);
    } else if (position > line_thresholds_.back()) {
      position -= line_thresholds_.back();
      double step = line_distance_ / 2;
      return - ceil(position / step);
    } else {
      return 9 - (std::lower_bound(line_thresholds_.begin(),
                                   line_thresholds_.end(),
                                   position) - line_thresholds_.begin());
    }
  }

 private:
  Mat src_;
  Mat gray_;
  Mat horizontal_;
  Mat vertical_;
  Mat notes_;
  int cropped_notes_ = 0;
  Classifier* classifier_;

  vector<int> line_positions_;

  vector<double> line_thresholds_;
  double line_distance_;
};

class Visualization {
 public:
  Visualization(const Line& line) : line_(line) {}
  void Save(const string& filename);

 public:
  string SymbolEntry(const Symbol& s);
  Line line_;
  static const string kHtmlTemplate;
};

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];

  string filename = argv[5];

  string action = "LINE";

  if (action == "LINE") {
    Classifier classifier(model_file, trained_file, mean_file, label_file);
    LineReader reader(&classifier, filename);
    auto line = reader.Read();
    line.Print();
    Visualization v(line);
    v.Save("results/test.html");
  } else {
    Classifier classifier(model_file, trained_file, mean_file, label_file);

    std::cout << "---------- Prediction for "
              << filename << " ----------" << std::endl;

    cv::Mat img = cv::imread(filename, -1);
    CHECK(!img.empty()) << "Unable to decode image " << filename;
    std::vector<Prediction> predictions = classifier.Classify(img);

    /* Print the top N predictions. */
    for (size_t i = 0; i < predictions.size(); ++i) {
      Prediction p = predictions[i];
      std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
                << p.first << "\"" << std::endl;
    }
  }
}

//////////////////////////////////////////////////
// Visualization
//////////////////////////////////////////////////

void Visualization::Save(const string& filename) {
  FILE* file = fopen(filename.c_str(), "w");

  vector<string> notes;
  vector<int> beams;

  string content;

  string current_notes;
  int current_group = -1;

  for (int i = 0; i < line_.symbols().size(); i++) {
    auto& symbol = line_.symbols()[i];

    auto entry = SymbolEntry(symbol);
    if (entry.size() == 0) continue;
    int group = line_.groups()[i];

    if (group != current_group) {
      if (current_notes.size() > 0)
        notes.push_back(current_notes);
      current_notes = "";

      current_group = group;

      if (group != -1)
        beams.push_back(notes.size());
    }

    if (current_notes.size() != 0) current_notes += ", ";
    current_notes += "\n" + entry;
  }

  if (current_notes.size() > 0) notes.push_back(current_notes);

  string concat = "notes = notes0";
  for (int i = 1; i < notes.size(); i++) concat += ".concat(notes" + std::to_string(i) + ")";
  concat += ";\n";

  for (int i = 0 ; i < notes.size(); i++) content += "notes" + std::to_string(i) + " = [" + notes[i] + "];\n";
  content += concat;
  for (int i = 0; i < beams.size(); i++)
    content += "var beam" + std::to_string(i) + " = new Vex.Flow.Beam(notes" + std::to_string(beams[i]) + ");\n";

  content += "Vex.Flow.Formatter.FormatAndDraw(ctx, stave, notes);\n";
  for (int i = 0; i < beams.size(); i++) content += "beam" + std::to_string(i) + ".setContext(ctx).draw();";

  fprintf(file, kHtmlTemplate.c_str(), content.c_str());
  fclose(file);
}

string Visualization::SymbolEntry(const Symbol& symbol) {
  const auto& type = symbol.type;
  if (type == N1 || type == N2 || type == N4 || type == N8) {
    //string pitch = "c/4";
    string pitch = symbol.position.ToKeyAndOctave();
    string duration = std::to_string(1 << int(type));
    return "new Vex.Flow.StaveNote({ keys: [\"" + pitch + "\"], duration: \"" + duration + "\" })";
  }

  if (type == R1 || type == R2 || type == R4 || type == N8) {
    string pitch = "b/4";
    string duration = std::to_string(1 << (int(type) - 4)) + "r";
    return "new Vex.Flow.StaveNote({ keys: [\"" + pitch + "\"], duration: \"" + duration + "\" })";
  }

  if (type == TC || type == BC) {
    return "";
  }

  if (type == BAR) {
    return "new Vex.Flow.BarNote()";
  }

  if (type == FA || type == SA) {
    return "";
  }
  assert(false && "Unimplemented type");
  return "";
}

const string Visualization::kHtmlTemplate = "<!DOCTYPE html><html><head><meta charset=\"UTF-8\"><title>Score</title><script src=\"https://ajax.googleapis.com/ajax/libs/jquery/1.11.3/jquery.min.js\"></script><script src=\"libs/vexflow-min.js\"></script><script>$(document).ready(function(){var canvas = $(\"canvas\")[0];var renderer = new Vex.Flow.Renderer(canvas,Vex.Flow.Renderer.Backends.CANVAS);var ctx = renderer.getContext();var stave = new Vex.Flow.Stave(10, 0, 1400);stave.addClef(\"treble\").setContext(ctx).draw();\n %s \n;Vex.Flow.Formatter.FormatAndDraw(ctx, stave, notes);})</script></head><body><canvas width=1400 height=100></canvas></body></html>";


//////////////////////////////////////////////////
// Classificationn
//////////////////////////////////////////////////

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->ForwardPrefilled();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

