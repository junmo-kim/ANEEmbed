import Foundation
let dir=URL(fileURLWithPath:CommandLine.arguments[1])
for (key,width,limit) in [("text",32,2048),("vision",18,2048)] {
  var values=[UInt16]()
  for index in 0..<limit {
    for dim in 0..<width {
      let freq = key=="text" ? 1/powf(10_000_000,Float(2*dim)/64) : powf(10000,-Float(dim)/18)
      let angle=Float(index)*freq
      values += [Float16(cosf(angle)).bitPattern.littleEndian,Float16(sinf(angle)).bitPattern.littleEndian]
    }
  }
  try values.withUnsafeBytes { try Data($0).write(to:dir.appendingPathComponent(key+"-rotary.fp16")) }
}
