"""Run all four tasks one after another:  python run_all.py"""
import traceback
import task1_restoration_segmentation as t1
import task2_feature_extraction as t2
import task3_object_recognition as t3
import task4_motion_estimation as t4

for name, mod in [("TASK 1", t1), ("TASK 2", t2), ("TASK 3", t3), ("TASK 4", t4)]:
    print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)
    try:
        mod.main()
    except Exception:
        print(f"{name} failed:")
        traceback.print_exc()
print("\nAll done. Check the outputs/ folder.")
