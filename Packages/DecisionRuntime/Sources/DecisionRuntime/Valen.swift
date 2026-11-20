import Foundation

@available(macOS 26.0, *)
extension Session {
  func valenAnswer(_ chunks: [Buffer], branch: Branch) async throws -> DecisionAnswer {
    guard let pool = manifest.rolePool, let mixer = manifest.roleMixer,
      (1...256).contains(branch.candidates.count), branch.roles.count == branch.candidates.count + 3
    else { throw DecisionError("Missing Valen role layout") }
    if manifest.batchedValenScore {
      return try await batchedValenAnswer(chunks, branch: branch, pool: pool, score: mixer)
    }
    guard let head = manifest.decisionHead else { throw DecisionError("Missing Valen decision head") }
    let width = manifest.width, hidden = manifest.hidden
    let poolFunction = try await backbone.function("valenPool", in: pool.asset)
    var pooled = [Buffer]()
    for span in branch.roles {
      guard !span.isEmpty, span.lowerBound >= 0, span.upperBound <= chunks.count * width else {
        throw DecisionError("Invalid Valen role span")
      }
      var length = Buffer.zeros([1, 1, 1, 1])
      length.write { p, _ in p[0] = Float16(span.count) }
      var previous = Buffer.zeros([1, hidden, 1, 1])
      for index in (span.lowerBound / width)...((span.upperBound - 1) / width) {
        var mask = Buffer.zeros([1, 1, 1, width])
        mask.write { p, s in
          for t in 0..<width where span.contains(index * width + t) { p[t * s[3]] = 1 }
        }
        var result = try await poolFunction.evaluate([
          "hidden": chunks[index], "valid": mask, "length": length, "previous": previous,
        ])
        previous = try result.tensor("pooled")
        guard previous.shape == [1, hidden, 1, 1] else { throw DecisionError("Invalid pooled role shape") }
      }
      pooled.append(previous)
    }
    let mix = try await backbone.function("valenMixer", in: mixer.asset)
    var logits = Buffer.zeros([1, 256, 1, 1])
    var mask = Buffer.zeros([1, 256, 1, 1])
    for i in branch.candidates.indices {
      var roles = Buffer.zeros([1, hidden, 1, 4])
      for (lane, index) in [0, 1, 2 + i, pooled.count - 1].enumerated() {
        pooled[index].read { a, sa in
          roles.write { b, sb in
            for c in 0..<hidden { b[c * sb[1] + lane * sb[3]] = a[c * sa[1]] }
          }
        }
      }
      var result = try await mix.evaluate(["pooled": roles])
      let value = try result.tensor("logit")
      guard value.shape == [1, 1, 1, 1] else { throw DecisionError("Invalid Valen logit shape") }
      value.read { a, _ in logits.write { b, s in b[i * s[1]] = a[0] } }
    }
    mask.write { p, s in for i in branch.candidates.count..<256 { p[i * s[1]] = -30000 } }
    let readout = try await backbone.function("readout", in: head.asset)
    var result = try await readout.evaluate(["logits": logits, "mask": mask])
    return try await formatDeviceAnswer(result.tensor("probabilities"), branch: branch)
  }

  private func batchedValenAnswer(_ chunks: [Buffer], branch: Branch, pool: Manifest.Asset,
                                  score: Manifest.Asset) async throws -> DecisionAnswer {
    let width = manifest.width, hidden = manifest.hidden
    let spans = branch.roles
    guard spans.allSatisfy({ !$0.isEmpty && $0.lowerBound >= 0 && $0.upperBound <= chunks.count * width }) else {
      throw DecisionError("Invalid Valen role span")
    }
    let poolFunction = try await backbone.function("valenPool288", in: pool.asset)
    var lengths = Buffer.zeros([1, 1, 1, 288])
    lengths.write { p, s in
      for (role, span) in spans.enumerated() { p[role * s[3]] = Float16(span.count) }
    }
    var pooled = Buffer.zeros([1, hidden, 1, 288])
    for (index, chunk) in chunks.enumerated() {
      let valid: Buffer = ValenRoleBatch.mask(spans: spans, chunk: index, width: width)
      var result = try await poolFunction.evaluate([
        "hidden": chunk, "valid": valid, "length": lengths, "previous": pooled,
      ])
      pooled = try result.tensor("pooled")
      guard pooled.shape == [1, hidden, 1, 288] else { throw DecisionError("Invalid batched Valen pool shape") }
    }
    let count = branch.candidates.count
    var mask = Buffer.zeros([1, 256, 1, 1])
    mask.write { p, s in for i in count..<256 { p[i * s[1]] = -30000 } }
    var inputs = statsCalibration(count)
    inputs["pooled"] = ValenRoleBatch.candidateRoles(pooled, count: count)
    inputs["mask"] = mask
    let scoreFunction = try await backbone.function("valenScore", in: score.asset)
    var result = try await scoreFunction.evaluate(inputs)
    let probabilities = try result.tensor("probabilities")
    let statistics = try result.tensor("statistics")
    return try await formatDeviceAnswer(probabilities, branch: branch, statistics: statistics)
  }
}

enum ValenRoleBatch {
  static func mask<B: DecisionBuffer>(spans: [Range<Int>], chunk: Int, width: Int) -> B {
    let offset = chunk * width
    var mask = B.zeros([1, 1, width, 288])
    mask.write { p, s in
      for (role, span) in spans.enumerated() {
        let part = span.clamped(to: offset..<(offset + width))
        for token in part { p[(token - offset) * s[2] + role * s[3]] = 1 }
      }
    }
    return mask
  }

  static func candidateRoles<B: DecisionBuffer>(_ pooled: B, count: Int) -> B {
    let hidden = pooled.shape[1]
    var packed = B.zeros([1, hidden, 4, 256])
    pooled.read { src, ss in
      packed.write { dst, ds in
        for channel in 0..<hidden {
          for candidate in 0..<count {
            for (role, column) in [0, 1, 2 + candidate, count + 2].enumerated() {
              dst[channel * ds[1] + role * ds[2] + candidate * ds[3]] = src[channel * ss[1] + column * ss[3]]
            }
          }
        }
      }
    }
    return packed
  }
}
