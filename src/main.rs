use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::types::VectorType;
use inkwell::{AddressSpace, OptimizationLevel};
use std::error::Error;

/// Convenience type alias for the `sum` function.
///
/// Calling this is innately `unsafe` because there's no guarantee it doesn't
/// do `unsafe` operations internally.
type SumFunc = unsafe extern "C" fn(u64, u64, u64) -> u64;

struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    fn jit_compile_sum(&self) -> Option<JitFunction<SumFunc>> {
        let i64_type = self.context.i64_type();
        let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into(), i64_type.into()], false);
        let function = self.module.add_function("sum", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        let x = function.get_nth_param(0)?.into_int_value();
        let y = function.get_nth_param(1)?.into_int_value();
        let z = function.get_nth_param(2)?.into_int_value();

        let sum = self.builder.build_int_add(x, y, "sum");
        let sum = self.builder.build_int_add(sum, z, "sum");

        self.builder.build_return(Some(&sum));

        unsafe { self.execution_engine.get_function("sum").ok() }
    }

    fn sum(
        &self,
    ) -> Option<JitFunction<unsafe extern "C" fn(*const f64, *const f64, *mut f64) -> f64>> {
        let width = 4;

        let f64_4_ty = self.context.f64_type().vec_type(width);
        let f64_ptr_ty = self.context.f64_type().ptr_type(AddressSpace::Generic);
        let fn_type = self.context.f64_type().fn_type(
            &[f64_ptr_ty.into(), f64_ptr_ty.into(), f64_ptr_ty.into()],
            false,
        );
        let function = self.module.add_function("sum", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        let x = function.get_nth_param(0)?.into_pointer_value();
        let y = function.get_nth_param(1)?.into_pointer_value();
        let z = function.get_nth_param(2)?.into_pointer_value();

        let mut x_vals = vec![];
        let mut y_vals = vec![];
        for i in 0..width {
            let idx = self.context.i64_type().const_int(i as u64, false);
            let x_ptr = unsafe { self.builder.build_gep(x, &[idx], "gep") };
            let x_val = self.builder.build_load(x_ptr, "load");
            let y_ptr = unsafe { self.builder.build_gep(y, &[idx], "gep") };
            let y_val = self.builder.build_load(y_ptr, "load");
            x_vals.push((idx, x_val));
            y_vals.push((idx, y_val))
        }

        let mut z_x = f64_4_ty.const_zero();
        for (i, x) in x_vals {
            z_x = self.builder.build_insert_element(z_x, x, i, "insert");
        }

        let mut z_y = f64_4_ty.const_zero();
        for (i, x) in y_vals {
            z_y = self.builder.build_insert_element(z_y, x, i, "insert");
        }

        let add = self.builder.build_float_add(z_x, z_y, "vec_add");

        let mut add_elems = vec![];
        let mut a = None;
        for i in 0..width {
            let idx = self.context.i64_type().const_int(i as u64, false);
            let val = self.builder.build_extract_element(add, idx, "ext");
            a = Some(val);
            add_elems.push(val)
        }

        for (i, e) in add_elems.into_iter().enumerate() {
            let idx = self.context.i64_type().const_int(i as u64, false);
            let ptr = unsafe { self.builder.build_gep(z, &[idx], "gep") };
            self.builder.build_store(ptr, e);
        }

        self.builder.build_return(Some(&a.unwrap()));
        self.module.print_to_stderr();

        unsafe { self.execution_engine.get_function("sum").ok() }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::create();
    let module = context.create_module("sum");
    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::Aggressive)?;
    let codegen = CodeGen {
        context: &context,
        module,
        builder: context.create_builder(),
        execution_engine,
    };

    // let sum = codegen
    //     .jit_compile_sum()
    //     .ok_or("Unable to JIT compile `sum`")?;
    let sum2 = codegen.sum().ok_or("Unable to JIT compile `sum`")?;

    let x = [1f64, 2f64, 3f64, 4f64];
    let y = [1f64, 2f64, 3f64, 4f64];
    let mut z = y;

    unsafe {
        let _q = sum2.call(x.as_ptr(), y.as_ptr(), z.as_mut_ptr());
        assert_eq!(z[0], 2f64);
        assert_eq!(z[1], 4f64);
        assert_eq!(z[2], 6f64);
        assert_eq!(z[3], 8f64);
    }

    Ok(())
}
