cust är en cuda wrapper som vi kan kalla på med cust::init()?, Module::from_ptx(PTX, &[])?;, Stream::new(...)?;

cuda builder låter en kompilera ner till gpu kernel, kallar sedan på rust_codegen_nvvm som kompilerar till ptx kod som gpu kan läsa

cuda_std låter en kalla på info från ett nvidia kort, thread::index_1d(), #[kernel] kmr också härifrån

build.rs körs innan main.rs, kompilerar i det här fallet kernelkod till ptx, pekar ut vilken mapp som har kernelkod och vart den ska lägga den kompilerade ptx koden



