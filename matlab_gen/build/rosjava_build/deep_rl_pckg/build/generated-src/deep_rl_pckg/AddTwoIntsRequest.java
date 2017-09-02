package deep_rl_pckg;

public interface AddTwoIntsRequest extends org.ros.internal.message.Message {
  static final java.lang.String _TYPE = "deep_rl_pckg/AddTwoIntsRequest";
  static final java.lang.String _DEFINITION = "std_msgs/Int64 a\nstd_msgs/Int64 b\n";
  std_msgs.Int64 getA();
  void setA(std_msgs.Int64 value);
  std_msgs.Int64 getB();
  void setB(std_msgs.Int64 value);
}
