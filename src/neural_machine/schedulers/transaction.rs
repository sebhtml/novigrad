use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use crate::{stream::DeviceStream, streams::stream::Stream, tensor::Error, Device, Instruction};

use super::StreamEventHandler;

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub enum Access {
    Read,
    Write,
}

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub struct Transaction {
    pub instruction: usize,
    pub operand: usize,
    pub access: Access,
}

pub fn get_instruction_transactions(
    instruction: usize,
    inputs: &[usize],
    outputs: &[usize],
) -> Vec<Transaction> {
    let mut transactions = vec![];
    for operand in inputs {
        let transaction = Transaction {
            instruction,
            operand: *operand,
            access: Access::Read,
        };
        transactions.push(transaction);
    }
    for operand in outputs {
        let transaction = Transaction {
            instruction,
            operand: *operand,
            access: Access::Write,
        };
        transactions.push(transaction);
    }
    transactions
}

// Example: for each read, find the prior write.
// Basically there are read accesses and write accesses.
// Here are the 4 pillars of the memory model:
// - a read has a prior write and it must remain the same. Changing the prior write makes the result incorrect.
// - a write has a prior write and it must remain the same. Changing the prior write makes the result incorrect.
// - a write has a prior read and it must remain the same. Changing the prior read makes the result incorrect.
// - a read has a prior read and it can change. Changing the prior read is allowed.
//        Example, if instructions 1, 2, 3 read operand 44, all those orderings are valid ones:
//           - 1, 2, 3
//           - 3, 2, 1
//           - 2, 1, 3
//           - ...
//       If we have 12 attention heads, that means that we can have 12 concurrent streams.
#[allow(unused)]
pub fn get_operand_transaction_pairs(
    access: &Access,
    prior_access: &Access,
    transactions: &[Transaction],
) -> BTreeMap<usize, Vec<(Transaction, Transaction)>> {
    // Group transactions per operand.
    let operand_transactions = group_by_operand(transactions);
    // For each read of an operand, find the most recent write before it­.
    let mut operand_pairs = BTreeMap::<usize, Vec<(Transaction, Transaction)>>::new();
    for (operand, transactions) in operand_transactions.iter() {
        for i in 0..transactions.len() {
            let transaction_i = &transactions[i];
            if &transaction_i.access == access {
                // Find the most recent write to this operand that happened in the past.
                for j in (0..i).rev() {
                    let transaction_j = &transactions[j];
                    if &transaction_j.access == prior_access {
                        let pair = (transaction_i.to_owned(), transaction_j.to_owned());
                        operand_pairs.entry(*operand).or_default().push(pair);
                        break;
                    }
                }
            }
        }
    }
    for (_, pairs) in operand_pairs.iter_mut() {
        // The tests use == so sorting makes the tests pass.
        pairs.sort();
    }
    operand_pairs
}

#[allow(unused)]
pub fn group_by_operand(transactions: &[Transaction]) -> BTreeMap<usize, Vec<Transaction>> {
    let mut operand_transactions = BTreeMap::<usize, Vec<Transaction>>::new();
    for transaction in transactions.iter() {
        let operand = transaction.operand;
        operand_transactions
            .entry(operand)
            .or_default()
            .push(transaction.to_owned());
    }
    operand_transactions
}

#[allow(unused)]
pub fn get_all_instruction_transactions(
    instructions: &[(Vec<usize>, Vec<usize>)],
) -> Vec<Transaction> {
    let mut transactions = vec![];
    for (instruction, (inputs, outputs)) in instructions.iter().enumerate() {
        let mut operand_transactions = get_instruction_transactions(instruction, inputs, outputs);
        transactions.extend_from_slice(&mut operand_transactions);
    }
    transactions
}

#[derive(Clone)]
pub struct TransactionEmitter {
    simple_instructions: Arc<Vec<(Vec<usize>, Vec<usize>)>>,
    pub actual_transactions: Arc<Mutex<Vec<Transaction>>>,
}

impl TransactionEmitter {
    pub fn new(simple_instructions: &Arc<Vec<(Vec<usize>, Vec<usize>)>>) -> Self {
        Self {
            simple_instructions: simple_instructions.clone(),
            actual_transactions: Default::default(),
        }
    }
}

impl StreamEventHandler for TransactionEmitter {
    fn on_execute(
        &mut self,
        streams: &Arc<Vec<Stream>>,
        _instructions: &Arc<Vec<Instruction>>,
        stream: usize,
        _device: &Device,
        _device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let stream_instructions = &streams[stream].instructions;
        for instruction in stream_instructions.iter() {
            let instruction = *instruction;
            let (inputs, outputs) = &self.simple_instructions[instruction];
            let mut instruction_transactions =
                get_instruction_transactions(instruction, inputs, outputs);
            self.actual_transactions
                .lock()
                .unwrap()
                .extend_from_slice(&mut instruction_transactions);
        }
        Ok(())
    }
}
