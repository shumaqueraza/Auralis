from auralis import Detector


def on_result(result):
    print(f"{result.class_name}: {result.confidence:.3f}")


detector = Detector(confidence=0.85)
detector.start(on_result)



try:
    input("Auralis is listening. Press Enter to stop...\n")
finally:
    detector.stop()
