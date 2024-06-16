use crate::schedulers::{
    transaction::Access,
    verification::{
        helper_all_instructions_are_executed_in_each_scheduler_execution,
        helper_all_instructions_are_executed_with_out_of_order_execution,
        helper_test_that_accesses_are_not_reordered,
    },
};

use super::scheduler::CpuStreamScheduler;

#[test]
fn test_reads_and_writes_of_same_operand_are_not_reordered() {
    let access = Access::Read;
    let prior_access = Access::Write;
    helper_test_that_accesses_are_not_reordered::<CpuStreamScheduler<_>>(access, prior_access);
}

#[test]
fn test_writes_and_writes_of_same_operand_are_not_reordered() {
    let access = Access::Write;
    let prior_access = Access::Write;
    helper_test_that_accesses_are_not_reordered::<CpuStreamScheduler<_>>(access, prior_access);
}

#[test]
fn test_writes_and_reads_of_same_operand_are_not_reordered() {
    let access = Access::Write;
    let prior_access = Access::Read;
    helper_test_that_accesses_are_not_reordered::<CpuStreamScheduler<_>>(access, prior_access);
}

#[test]
fn test_all_instructions_are_executed_with_out_of_order_execution() {
    helper_all_instructions_are_executed_with_out_of_order_execution::<CpuStreamScheduler<_>>();
}

#[test]
fn test_all_instructions_are_executed_in_each_scheduler_execution() {
    helper_all_instructions_are_executed_in_each_scheduler_execution::<CpuStreamScheduler<_>>();
}
