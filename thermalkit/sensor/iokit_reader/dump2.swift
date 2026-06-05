import Foundation
import IOKit.hid

// IOHIDEventSystemClientCreateSimpleClient IS in public headers — use it directly
// Other functions need @_silgen_name
@_silgen_name("IOHIDEventSystemClientSetMatching")
func IOHIDEventSystemClientSetMatching(_ client: IOHIDEventSystemClient, _ matching: CFDictionary)

@_silgen_name("IOHIDEventSystemClientCopyServices")
func IOHIDEventSystemClientCopyServices(_ client: IOHIDEventSystemClient) -> NSArray

@_silgen_name("IOHIDServiceClientCopyProperty")
func IOHIDServiceClientCopyProperty(_ service: IOHIDServiceClient, _ key: CFString) -> CFTypeRef?

@_silgen_name("IOHIDServiceClientCopyEvent")
func IOHIDServiceClientCopyEvent(_ service: IOHIDServiceClient, _ type: Int32, _ options: Int32, _ timeout: Int32) -> IOHIDEvent?

@_silgen_name("IOHIDEventGetFloatValue")
func IOHIDEventGetFloatValue(_ event: IOHIDEvent, _ field: Int32) -> Double

let client = IOHIDEventSystemClientCreateSimpleClient(kCFAllocatorDefault)
let matching: [String: Any] = ["PrimaryUsagePage": 0xFF00, "PrimaryUsage": 5]
IOHIDEventSystemClientSetMatching(client, matching as CFDictionary)

let services = IOHIDEventSystemClientCopyServices(client) as! [IOHIDServiceClient]
print("Found \(services.count) services")

var seen = Set<String>()
for service in services {
    guard let name = IOHIDServiceClientCopyProperty(service, "Product" as CFString) as? String,
          !seen.contains(name) else { continue }
    seen.insert(name)
    if let event = IOHIDServiceClientCopyEvent(service, 15, 0, 0) {
        let t = IOHIDEventGetFloatValue(event, (15 << 16) | 0)
        print(String(format: "  %-30s %.2f C", name, t))
    } else {
        print("  \(name): no event")
    }
}
