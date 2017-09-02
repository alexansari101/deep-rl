package deep_rl_pckg;

public interface AddTwoIntsResponse extends org.ros.internal.message.Message {
  static final java.lang.String _TYPE = "deep_rl_pckg/AddTwoIntsResponse";
  static final java.lang.String _DEFINITION = "std_msgs/Int64 sum";
  std_msgs.Int64 getSum();
  void setSum(std_msgs.Int64 value);
}
