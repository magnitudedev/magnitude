// The desktop tray addon (macOS): the tray menu's live model status row, a native view on one item
// of the tray menu Electron builds, so the row updates while the menu is open.
//
//   <model name> · <phase>        <spinner | 58% ◔ | 18.4 GB>
//
// Like a plain menu item, the row is as wide as the full model name needs and the menu widens to
// fit. Only the model changes its width: it holds room for the widest phase and detail, and an
// open menu never narrows.

#import <AppKit/AppKit.h>
#include <math.h>
#include <node_api.h>

static const CGFloat rowHeight = 22;
/// Horizontal inset of menu item text.
static const CGFloat textInset = 14;
static const CGFloat phaseGap = 4;
static const CGFloat detailGap = 8;
static const CGFloat ringGap = 5;
static const CGFloat ringDiameter = 12;
static const CGFloat ringLineWidth = 2;
/// The widest memory figure the detail shows.
static NSString *const widestMemory = @"888.8 GB";

/// Load progress as a ring: a filling arc when measured, a spinning arc when not.
@interface MagnitudeProgressRing : NSView
@property(nonatomic) double fraction;
@property(nonatomic) BOOL spinning;
@end

@implementation MagnitudeProgressRing {
  NSTimer *_timer;
  CGFloat _angle;
}

- (void)setFraction:(double)fraction {
  _fraction = fraction;
  self.needsDisplay = YES;
}

- (void)setSpinning:(BOOL)spinning {
  if (_spinning == spinning) return;
  _spinning = spinning;
  [_timer invalidate];
  _timer = nil;
  if (spinning) {
    __weak MagnitudeProgressRing *weakSelf = self;
    _timer = [NSTimer timerWithTimeInterval:1.0 / 60 repeats:YES block:^(NSTimer *timer) {
      (void)timer;
      MagnitudeProgressRing *ring = weakSelf;
      if (!ring) return;
      ring->_angle = fmod(ring->_angle + 6, 360);
      ring.needsDisplay = YES;
    }];
    // Common modes include the open menu's event-tracking mode.
    [NSRunLoop.mainRunLoop addTimer:_timer forMode:NSRunLoopCommonModes];
  }
  self.needsDisplay = YES;
}

- (void)drawRect:(NSRect)dirtyRect {
  (void)dirtyRect;
  NSPoint centre = NSMakePoint(NSMidX(self.bounds), NSMidY(self.bounds));
  CGFloat radius = (MIN(NSWidth(self.bounds), NSHeight(self.bounds)) - ringLineWidth) / 2;
  NSBezierPath *track = [NSBezierPath bezierPath];
  [track appendBezierPathWithArcWithCenter:centre radius:radius startAngle:0 endAngle:360];
  track.lineWidth = ringLineWidth;
  [[NSColor.labelColor colorWithAlphaComponent:0.15] setStroke];
  [track stroke];
  CGFloat start = self.spinning ? 90 - _angle : 90;
  CGFloat sweep = self.spinning ? 90 : 360 * self.fraction;
  if (sweep <= 0) return;
  NSBezierPath *arc = [NSBezierPath bezierPath];
  [arc appendBezierPathWithArcWithCenter:centre radius:radius startAngle:start endAngle:start - sweep clockwise:YES];
  arc.lineWidth = ringLineWidth;
  arc.lineCapStyle = NSLineCapStyleRound;
  [NSColor.controlAccentColor setStroke];
  [arc stroke];
}
@end

static NSInteger expectedIndex = -1;
static id menuObserver;
static NSArray<NSString *> *phaseWords;
static NSView *statusView;
static NSTextField *modelName;
static NSTextField *phase;
/// The percentage beside the ring, or the memory in use once loaded.
static NSTextField *detail;
static MagnitudeProgressRing *ring;
static NSLayoutConstraint *detailBesideRing;
static NSLayoutConstraint *detailAtEdge;
/// The row's width beside the model name: insets, the widest phase and the widest detail.
static CGFloat besideName;

static NSString *currentModel = @"";
static NSString *currentPhase = @"";
/// A measured fraction, or negative when the phase has none.
static double currentFraction = -1;
/// Text in place of progress (the memory in use), or nil.
static NSString *currentText = nil;

static NSTextField *menuLabel(NSFont *font) {
  NSTextField *label = [NSTextField labelWithString:@""];
  label.font = font;
  label.textColor = NSColor.disabledControlTextColor;
  label.translatesAutoresizingMaskIntoConstraints = NO;
  return label;
}

static CGFloat textWidth(NSString *text, NSFont *font) {
  return ceil([text sizeWithAttributes:@{NSFontAttributeName: font}].width);
}

/// The width the full model name needs.
static CGFloat fittedWidth(void) {
  return textWidth(currentModel, modelName.font) + besideName;
}

static void apply(void) {
  if (![modelName.stringValue isEqualToString:currentModel]) {
    modelName.stringValue = currentModel;
    // An open menu keeps its width, so the row never narrows while shown; the next open fits it.
    [statusView setFrameSize:NSMakeSize(MAX(fittedWidth(), NSWidth(statusView.frame)), rowHeight)];
  }
  phase.stringValue = [@"· " stringByAppendingString:currentPhase];
  BOOL text = currentText != nil;
  BOOL measured = currentFraction >= 0;
  ring.hidden = text;
  ring.spinning = !text && !measured;
  if (measured) ring.fraction = currentFraction;
  detail.hidden = !text && !measured;
  detail.stringValue = text ? currentText
                     : measured ? [NSString stringWithFormat:@"%d%%", (int)floor(currentFraction * 100)]
                                : @"";
  // Deactivate before activating, so the two placements never hold at once.
  detailBesideRing.active = NO;
  detailAtEdge.active = NO;
  (text ? detailAtEdge : detailBesideRing).active = YES;
}

static void buildView(void) {
  NSFont *font = [NSFont menuFontOfSize:0];
  modelName = menuLabel(font);
  phase = menuLabel(font);
  detail = menuLabel([NSFont monospacedDigitSystemFontOfSize:font.pointSize weight:NSFontWeightRegular]);
  detail.alignment = NSTextAlignmentRight;
  ring = [[MagnitudeProgressRing alloc] init];
  ring.translatesAutoresizingMaskIntoConstraints = NO;
  statusView = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 0, rowHeight)];
  statusView.autoresizingMask = NSViewWidthSizable;
  for (NSView *view in @[modelName, phase, detail, ring]) [statusView addSubview:view];
  for (NSTextField *label in @[phase, detail]) {
    [label setContentCompressionResistancePriority:NSLayoutPriorityRequired
                                    forOrientation:NSLayoutConstraintOrientationHorizontal];
    [label setContentHuggingPriority:NSLayoutPriorityRequired forOrientation:NSLayoutConstraintOrientationHorizontal];
  }
  CGFloat widestPhase = 0;
  for (NSString *word in phaseWords) {
    widestPhase = MAX(widestPhase, textWidth([@"· " stringByAppendingString:word], font));
  }
  CGFloat percentWidth = textWidth(@"100%", detail.font) + 2;
  CGFloat widestDetail = MAX(percentWidth + ringGap + ringDiameter, textWidth(widestMemory, detail.font));
  besideName = 2 * textInset + phaseGap + widestPhase + detailGap + widestDetail;
  [NSLayoutConstraint activateConstraints:@[
    [modelName.leadingAnchor constraintEqualToAnchor:statusView.leadingAnchor constant:textInset],
    [modelName.centerYAnchor constraintEqualToAnchor:statusView.centerYAnchor],
    [phase.leadingAnchor constraintEqualToAnchor:modelName.trailingAnchor constant:phaseGap],
    [phase.firstBaselineAnchor constraintEqualToAnchor:modelName.firstBaselineAnchor],
    [ring.trailingAnchor constraintEqualToAnchor:statusView.trailingAnchor constant:-textInset],
    [ring.widthAnchor constraintEqualToConstant:ringDiameter],
    [ring.heightAnchor constraintEqualToConstant:ringDiameter],
    [ring.centerYAnchor constraintEqualToAnchor:statusView.centerYAnchor],
    [detail.firstBaselineAnchor constraintEqualToAnchor:modelName.firstBaselineAnchor],
  ]];
  detailBesideRing = [detail.trailingAnchor constraintEqualToAnchor:ring.leadingAnchor constant:-ringGap];
  detailAtEdge = [detail.trailingAnchor constraintEqualToAnchor:statusView.trailingAnchor constant:-textInset];
  apply();
}

static NSString *stringArgument(napi_env env, napi_value value) {
  size_t length;
  if (napi_get_value_string_utf8(env, value, NULL, 0, &length) != napi_ok) return nil;
  char *buffer = malloc(length + 1);
  if (!buffer) return nil;
  napi_get_value_string_utf8(env, value, buffer, length + 1, &length);
  NSString *string = [NSString stringWithUTF8String:buffer];
  free(buffer);
  return string;
}

static void attach(NSNotification *note) {
  NSInteger added = [note.userInfo[@"NSMenuItemIndex"] integerValue];
  if (expectedIndex < 0 || added != expectedIndex) return;
  expectedIndex = -1;
  if (!statusView) buildView();
  [statusView removeFromSuperview];
  // Each open fits the current name, whatever an earlier open grew to.
  [statusView setFrameSize:NSMakeSize(fittedWidth(), rowHeight)];
  [(NSMenu *)note.object itemAtIndex:added].view = statusView;
}

/// configureTrayStatus(phases): the phase words the row reserves room for. Once, before the
/// first menu.
static napi_value Configure(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1];
  uint32_t count;
  if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc != 1 ||
      napi_get_array_length(env, argv[0], &count) != napi_ok) {
    napi_throw_type_error(env, NULL, "configureTrayStatus expects the phase words");
    return NULL;
  }
  NSMutableArray<NSString *> *words = [NSMutableArray arrayWithCapacity:count];
  for (uint32_t index = 0; index < count; index++) {
    napi_value element;
    NSString *word = napi_get_element(env, argv[0], index, &element) == napi_ok ? stringArgument(env, element) : nil;
    if (!word) {
      napi_throw_type_error(env, NULL, "configureTrayStatus expects the phase words");
      return NULL;
    }
    [words addObject:word];
  }
  phaseWords = words;
  if (!menuObserver) {
    menuObserver = [NSNotificationCenter.defaultCenter addObserverForName:NSMenuDidAddItemNotification
                                                                   object:nil
                                                                    queue:nil
                                                               usingBlock:^(NSNotification *note) { attach(note); }];
  }
  return NULL;
}

/// expectTrayStatus(index): the next menu built carries the status row at `index`.
static napi_value Expect(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1];
  int32_t index;
  if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc != 1 ||
      napi_get_value_int32(env, argv[0], &index) != napi_ok || !phaseWords) {
    napi_throw_type_error(env, NULL, "expectTrayStatus expects an item index after configuration");
    return NULL;
  }
  expectedIndex = index;
  return NULL;
}

/// updateTrayStatus(model, phase, fraction | null, text | null): neither fraction nor text shows
/// the spinner.
static napi_value Update(napi_env env, napi_callback_info info) {
  size_t argc = 4;
  napi_value argv[4];
  napi_valuetype fractionType, textType;
  NSString *model = nil, *word = nil;
  if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) == napi_ok && argc == 4) {
    model = stringArgument(env, argv[0]);
    word = stringArgument(env, argv[1]);
  }
  if (!model || !word || napi_typeof(env, argv[2], &fractionType) != napi_ok ||
      napi_typeof(env, argv[3], &textType) != napi_ok ||
      (fractionType != napi_number && fractionType != napi_null) ||
      (textType != napi_string && textType != napi_null)) {
    napi_throw_type_error(env, NULL, "updateTrayStatus expects a model, a phase, a fraction or null, and text or null");
    return NULL;
  }
  currentModel = model;
  currentPhase = word;
  currentFraction = -1;
  if (fractionType == napi_number) napi_get_value_double(env, argv[2], &currentFraction);
  currentText = textType == napi_string ? stringArgument(env, argv[3]) : nil;
  if (statusView) apply();
  return NULL;
}

NAPI_MODULE_INIT() {
  napi_property_descriptor methods[] = {
    {"configureTrayStatus", NULL, Configure, NULL, NULL, NULL, napi_default, NULL},
    {"expectTrayStatus", NULL, Expect, NULL, NULL, NULL, napi_default, NULL},
    {"updateTrayStatus", NULL, Update, NULL, NULL, NULL, napi_default, NULL},
  };
  napi_define_properties(env, exports, sizeof(methods) / sizeof(methods[0]), methods);
  return exports;
}
