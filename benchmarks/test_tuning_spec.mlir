// Minimal no-op tuning spec for testing purposes.
// This spec does not modify any operations, just passes them through.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed}) -> !transform.any_op attributes {iree_codegen.tuning_spec_entrypoint} {
    transform.yield %arg0 : !transform.any_op
  }
}
